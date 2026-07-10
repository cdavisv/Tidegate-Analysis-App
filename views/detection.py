"""Image Detection page: images -> species dataset (demo / MegaDetector / GPT)."""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import plotly.express as px
import streamlit as st

import ui_common as ui
from vision import (
    detections_to_camera_df,
    detections_to_long_df,
    get_detector,
    summarize_detections,
)
from vision.image_source import collect_images, save_uploaded_files
from vision.pipeline import DETECTOR_LABELS

cfg = ui.get_config()

ui.app_header(
    "Image Detection",
    "Identify wildlife in camera-trap images and generate the camera dataset.",
    icon="📷",
)
ui.workflow_chips(active="detect")

# -------------------------------------------------------------------------
# 1. Choose a detector
# -------------------------------------------------------------------------
st.subheader("1 · Choose a detector")

labels = {
    "demo": DETECTOR_LABELS["demo"],
    "megadetector": DETECTOR_LABELS["megadetector"],
    "openai": DETECTOR_LABELS["openai"],
}
choice = st.radio(
    "Detection engine",
    options=list(labels.keys()),
    format_func=lambda k: labels[k],
    horizontal=True,
    help="The demo detector is synthetic and needs no models or keys — great for a dry run. "
    "MegaDetector runs locally (needs PyTorch + Pytorch-Wildlife). OpenAI uses a GPT vision model.",
)

detector_kwargs: dict = {}
detector = None
avail_ok, avail_reason = True, ""

if choice == "demo":
    with st.expander("Demo detector settings", expanded=False):
        prob = st.slider("Detection probability", 0.0, 1.0, 0.30, 0.05,
                         help="Chance a given image contains a detected animal.")
        seed = st.number_input("Random seed", value=1234, step=1)
    detector_kwargs = {"detection_probability": prob, "seed": int(seed)}
    st.info("Synthetic detector — results are fabricated for testing the pipeline, not real ecology.")

elif choice == "megadetector":
    col1, col2, col3 = st.columns(3)
    with col1:
        version = st.text_input("MegaDetector version", value=cfg.megadetector_version)
    with col2:
        det_thr = st.slider("Detection threshold", 0.05, 0.9, float(cfg.detection_threshold), 0.05)
    with col3:
        use_sn = st.checkbox("SpeciesNet classification", value=True,
                             help="Classify animal crops to species (requires the 'speciesnet' package).")
    detector_kwargs = {
        "version": version,
        "detection_threshold": det_thr,
        "use_speciesnet": use_sn,
        "country": "USA",
        "admin1_region": "OR",
    }
    detector = get_detector("megadetector", **detector_kwargs)
    avail_ok, avail_reason = detector.available()
    if avail_ok:
        st.success("MegaDetector is installed and ready.")
    else:
        st.warning(f"MegaDetector not ready: {avail_reason}")
        st.markdown(
            "Install the local CV stack on your machine (ideally with a GPU):\n"
            "```bash\npip install PytorchWildlife    # MegaDetector v5/v6\n"
            "pip install speciesnet          # optional species classifier\n```\n"
            "See the AddaxAI project for a packaged desktop workflow: "
            "https://github.com/PetervanLunteren/AddaxAI"
        )

elif choice == "openai":
    col1, col2 = st.columns([2, 1])
    with col1:
        model = st.text_input("OpenAI model", value=cfg.openai_vision_model,
                              help="Any vision-capable OpenAI model string (e.g. gpt-5.5).")
    with col2:
        max_px = st.number_input("Max image size (px)", value=1024, step=128,
                                 help="Images are downscaled before upload to control cost.")
    key_default = cfg.openai_api_key or ""
    api_key = st.text_input(
        "OpenAI API key",
        value=key_default,
        type="password",
        help="Used only for this session; read from OPENAI_API_KEY if left set. Never stored to disk.",
    )
    detector_kwargs = {"model": model, "api_key": api_key or None, "max_image_px": int(max_px)}
    detector = get_detector("openai", **detector_kwargs)
    avail_ok, avail_reason = detector.available()
    (st.success if avail_ok else st.warning)(
        "OpenAI detector ready." if avail_ok else f"Not ready: {avail_reason}"
    )
    st.caption("Cost note: one API call per image. Use the image cap below for large sets.")

st.divider()

# -------------------------------------------------------------------------
# 2. Provide images
# -------------------------------------------------------------------------
st.subheader("2 · Provide images")
src_tab, up_tab, demo_tab = st.tabs(
    ["📁 Folder on disk", "⬆️ Upload images", "✨ Demo set (no files)"]
)

images = []
source_desc = ""

with src_tab:
    folder = st.text_input("Image folder path", placeholder="/path/to/camera_images",
                           help="A directory of camera-trap images on the machine running this app.")
    c1, c2 = st.columns(2)
    recursive = c1.checkbox("Include sub-folders", value=True)
    limit = c2.number_input("Max images (0 = all)", value=200, min_value=0, step=50,
                            help="Cap the number of images processed. Keep small for LLM runs.")
    if folder:
        if os.path.isdir(folder):
            found = collect_images(folder, recursive=recursive,
                                   limit=(int(limit) or None))
            st.caption(f"Found **{len(found)}** image(s).")
            images = found
            source_desc = f"{len(found)} images from folder"
        else:
            st.error("That folder was not found on this machine.")

with up_tab:
    uploads = st.file_uploader(
        "Upload images", type=["jpg", "jpeg", "png", "tif", "tiff", "bmp", "webp"],
        accept_multiple_files=True,
    )
    if uploads:
        tmpdir = os.path.join(tempfile.gettempdir(), "tidegate_uploads")
        paths = save_uploaded_files(uploads, tmpdir)
        images = collect_images(tmpdir, recursive=False, limit=None)
        # collect_images sorts the folder; restrict to just-uploaded files
        upl_names = {os.path.basename(p) for p in paths}
        images = [im for im in images if os.path.basename(im.path) in upl_names]
        st.caption(f"Uploaded **{len(images)}** image(s).")
        source_desc = f"{len(images)} uploaded images"

with demo_tab:
    st.caption(
        "No camera images handy? Generate a synthetic image set and run the **demo "
        "detector** to see the image → dataset step end-to-end — no files, models, "
        "or keys needed. Timestamps span the bundled Willanch sensor window, so the result "
        "lines up with the demo tide/weather data on the Analysis page."
    )
    dc1, dc2 = st.columns(2)
    demo_n = dc1.number_input("How many images", min_value=10, max_value=5000, value=200,
                              step=10, key="demo_n")
    demo_seed = dc2.number_input("Random seed", value=1234, step=1, key="demo_seed")
    if st.button("✨ Generate demo set & detect", type="primary", key="demo_run"):
        from vision.demo_data import synthetic_image_refs
        refs = synthetic_image_refs(int(demo_n), seed=int(demo_seed))
        demo_det = get_detector("demo", detection_probability=0.30, seed=int(demo_seed))
        with st.spinner(f"Detecting on {len(refs):,} synthetic images…"):
            dets = demo_det.detect_batch(refs)
        st.session_state[ui.K_DETECTIONS] = dets
        st.session_state[ui.K_CAMERA_DF] = detections_to_camera_df(dets)
        st.session_state[ui.K_CAMERA_SRC] = f"Demo (synthetic) · {len(refs):,} generated images"
        st.success(
            f"Generated and detected {len(dets):,} synthetic images. See the results below, "
            "then head to the Analysis page (the demo sensor loads there)."
        )

st.divider()

# -------------------------------------------------------------------------
# 3. Run detection
# -------------------------------------------------------------------------
st.subheader("3 · Run detection")

if detector is None:
    detector = get_detector(choice, **detector_kwargs)

run_disabled = (len(images) == 0) or (not avail_ok)
if len(images) == 0:
    st.caption("Provide a folder or upload images to enable detection.")

if st.button("Run detection", type="primary", disabled=run_disabled):
    progress_bar = st.progress(0.0, text="Starting…")

    def _progress(i, total, path):
        frac = (i / total) if total else 1.0
        name = os.path.basename(path) if path else "done"
        progress_bar.progress(min(frac, 1.0), text=f"{i}/{total} · {name}")

    try:
        with st.spinner("Detecting…"):
            detections = detector.detect_batch(images, progress=_progress)
        progress_bar.progress(1.0, text="Complete")
        camera_df = detections_to_camera_df(detections)
        st.session_state[ui.K_DETECTIONS] = detections
        st.session_state[ui.K_CAMERA_DF] = camera_df
        st.session_state[ui.K_CAMERA_SRC] = f"{labels[choice].split(' (')[0]} · {source_desc}"
        st.success(f"Detection complete on {len(detections)} image(s).")
    except Exception as exc:
        st.error(f"Detection failed: {exc}")

# -------------------------------------------------------------------------
# 4. Results
# -------------------------------------------------------------------------
detections = st.session_state.get(ui.K_DETECTIONS)
if detections:
    st.divider()
    st.subheader("Detection results")
    summ = summarize_detections(detections)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Images", f"{summ['n_images']:,}")
    m2.metric("With animals", f"{summ['n_with_animals']:,}")
    m3.metric("Detection rate", f"{summ['detection_rate']:.1f}%")
    m4.metric("Unique species", summ["unique_species"])
    if summ["n_errors"]:
        st.caption(f"⚠️ {summ['n_errors']} image(s) errored during detection.")

    if summ["species_counts"]:
        sc = pd.DataFrame(
            {"Species": list(summ["species_counts"].keys()),
             "Individuals": list(summ["species_counts"].values())}
        )
        fig = px.bar(sc, x="Individuals", y="Species", orientation="h",
                     title="Detected individuals by species",
                     color="Individuals", color_continuous_scale="Teal")
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=380,
                          margin=dict(l=10, r=10, t=50, b=10))
        st.plotly_chart(fig, use_container_width=True)

    animal_dets = [d for d in detections
                   if d.has_animal and d.image_path and os.path.exists(d.image_path)]
    if animal_dets:
        shown = animal_dets[:6]
        with st.expander(f"Preview detected frames ({len(shown)} of {len(animal_dets)})",
                         expanded=False):
            pcols = st.columns(3)
            for i, d in enumerate(shown):
                cap = ", ".join(f"{it.label}\u00d7{it.count}" for it in d.animals[:3])
                try:
                    pcols[i % 3].image(d.image_path, caption=cap, use_container_width=True)
                except Exception:
                    pcols[i % 3].caption(f"{os.path.basename(d.image_path)} - {cap}")

    with st.expander("Per-detection table", expanded=False):
        long_df = detections_to_long_df(detections)
        st.dataframe(long_df, use_container_width=True, height=320)

    camera_df = st.session_state.get(ui.K_CAMERA_DF)
    if camera_df is not None:
        st.download_button(
            "Download camera dataset CSV",
            data=camera_df.to_csv(index=False),
            file_name="camera_dataset_from_detection.csv",
            mime="text/csv",
        )
        st.info("This dataset is now available on the **Analysis** page.")
        ui.page_link("views/analysis.py", label="Go to Analysis →", icon=":material/insights:")

with st.expander("How detection feeds the analysis"):
    st.markdown(
        "Each image becomes one row of a wide-format *camera dataset* "
        "(`Species 1`, `Species 1 Count`, `DateTime`, …) — identical to a hand-labelled "
        "sheet. Images with no animal become valid *no-detection* records, which the "
        "dual-framework analysis needs to separate operational bias from behavior. "
        "Timestamps come from EXIF, then the filename, then file modification time."
    )
