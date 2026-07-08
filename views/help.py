"""In-app Help & Docs page."""

from __future__ import annotations

import streamlit as st

import ui_common as ui

ui.app_header("Help & Documentation", "Everything you need to run the full pipeline.", icon="❓")

t_over, t_data, t_detect, t_weather, t_method, t_config, t_faq = st.tabs(
    ["Overview", "Data formats", "Detectors", "Weather", "Methodology", "Config", "FAQ"]
)

with t_over:
    st.markdown(
        """
### The pipeline

1. **Image Detection** — Turn camera-trap images into a species dataset.
   Pick a detector, point it at a folder (or upload images), and run it. The output is a
   wide-format *camera dataset* identical to a hand-labelled sheet.
2. **Weather Import** — Fetch weather & tide data (Open-Meteo / NOAA / Synoptic) or upload a
   CSV, then merge it onto your sensor timeline.
3. **Analysis** — Combine camera + sensor data and run the dual-framework analysis to see how
   wildlife activity tracks tides, gate operations, and weather.

Each step hands its output to the next through the session (see the **Session data** panel in
the sidebar). You can also start at any step by uploading the relevant CSV.

**Fastest way to see it work:** Home → *Load Willanch demo data* → Analysis → *Run full analysis*.
        """
    )

with t_data:
    st.markdown(
        """
### Camera dataset (CSV)
One row per image/observation. Required columns:

- `DateTime` **or** both `Date` and `Time`.
- `Species 1` (blank when nothing was detected). Optional `Species 2`, `Species 3`, …
- Optional counts: `Species 1 Count` (or `Species Count 1`), and `Notes 1`, etc.

Rows with a blank `Species 1` are kept as **no-animal camera records** — essential for the
dual-framework analysis. The Image Detection page produces exactly this shape automatically.

### Water / tide / sensor dataset (CSV)
- `DateTime` (or `Date` + `Time`).
- Any of: `Gate Opening MTR [Degrees]`, `Gate Opening Top Hinge [Degrees]`,
  `Tidal Level Outside Tidegate [m]` (→ `Depth`), `Tidal Level Inside Tidegate [m]`,
  `Air Temp [C]`, `Wind Speed [km/h]` — plus any extra numeric weather columns, which are now
  **retained** and flow into the combined dataset.
- Columns already using internal names (`Gate_Opening_MTR_Deg`, `Depth`, `Air_Temp_C`, …) are
  used as-is. Zero depths are treated as sensor errors (→ NaN). A UTF-8 BOM on the header is
  handled automatically.

### Weather CSV (for upload)
Bracketed-unit headers like `Air Temp [C]`, `Wind Speed [km/h]`, `Precipitation [cm]`,
`Barometric Pressure [mbar]`, `Humidity [%]` are auto-detected and unit-converted.
        """
    )

with t_detect:
    st.markdown(
        """
### Detectors

**Demo (synthetic).** No dependencies. Fabricates deterministic, estuary-weighted detections so
you can exercise the whole pipeline without models or keys. *Not real ecology.*

**MegaDetector + SpeciesNet (local CV).** The AddaxAI-style path. MegaDetector localizes
animals/people/vehicles; SpeciesNet (optional) classifies species. Runs locally — a GPU helps.

```bash
pip install PytorchWildlife     # MegaDetector v5/v6 (Microsoft CameraTraps)
pip install speciesnet          # optional species classifier (Google)
```

Related projects by Peter van Lunteren: **AddaxAI** (desktop app around MegaDetector/SpeciesNet)
— https://github.com/PetervanLunteren/AddaxAI

**OpenAI GPT vision (LLM).** Sends each image to a multimodal GPT model that returns species +
counts as JSON. Configure the model string (default `gpt-5.5`) and paste your API key (used only
for the session; read from `OPENAI_API_KEY` if set). One API call per image — mind the cost on
large sets and use the image cap.

Timestamps are read from EXIF `DateTimeOriginal`, then the filename, then file modified time.
        """
    )

with t_weather:
    st.markdown(
        """
### Weather & tide sources (all free)

- **Open-Meteo** *(recommended)* — no API key. Historical archive + forecast. Pulls temperature,
  wind, precipitation, pressure, humidity, and solar radiation by lat/lon.
- **NOAA** — NWS station observations, plus **CO-OPS** tide water level when you provide a station
  id (e.g. `9432780`, Charleston OR).
- **Synoptic / MesoWest** — dense station network; needs a free token (`SYNOPTIC_TOKEN`).
- **CSV upload** — bring your own weather-station export.

All sources normalize to: `Air_Temp_C`, `Wind_Speed_km_h`, `Precipitation_cm`,
`Barometric_Pressure_mbar`, `Humidity_pct`, `Solar_Radiation_W_m2`, `Water_Level_m`.

Merging uses nearest-time matching within a tolerance window; choose whether weather only fills
gaps (`sensor`) or overwrites (`weather`).
        """
    )

with t_method:
    st.markdown(
        """
### Dual-framework methodology

- **Camera Activity Pattern Analysis** uses *all* time periods and measures when cameras were
  operational relative to conditions → equipment performance & monitoring bias.
- **Wildlife Detection Efficiency Analysis** restricts to camera-active periods and measures how
  often animals were detected → genuine wildlife behavior.

Comparing the two separates *how we watched* from *what the animals did*. Layered on top:

- **Tidal cycle & phase** — classifies rising/falling/slack and models a continuous phase
  (0 = low, 0.5 = high) to find peak-activity windows.
- **Gate combinations** — hypothesis tests across MTR/top-hinge positions.
- **Weather patterns** — detection rate across binned weather variables.

Statistics include chi-square tests and GLM modeling (see `analysis.py`).
        """
    )

with t_config:
    cfg = ui.get_config()
    st.markdown("### Configuration")
    st.markdown(
        "Defaults live in `config.json` (site name, coordinates, model names). Secrets come from "
        "the environment and are **never** written to disk:"
    )
    st.code(
        "OPENAI_API_KEY=sk-...          # OpenAI GPT vision\n"
        "OPENAI_VISION_MODEL=gpt-5.5    # override default model\n"
        "SYNOPTIC_TOKEN=...             # Synoptic weather\n"
        "TIDEGATE_LAT=43.45             # site latitude\n"
        "TIDEGATE_LON=-124.20           # site longitude\n"
        "TIDEGATE_TZ=America/Los_Angeles",
        language="bash",
    )
    st.markdown("**Current effective config:**")
    st.json(cfg.public_dict())

with t_faq:
    st.markdown(
        """
### Troubleshooting

**"MegaDetector not ready."** PyTorch / Pytorch-Wildlife aren't installed in this environment.
Install them (see Detectors) or use the Demo or OpenAI detector.

**"No OpenAI API key."** Paste a key on the Detection page or set `OPENAI_API_KEY`.

**Weather fetch failed.** The host may lack internet access, or the source may be rate-limited.
Try Open-Meteo, or upload a weather CSV instead.

**My analysis has no weather tab data.** Supply a sensor CSV that includes weather columns, or use
the Weather Import page to fetch and merge weather first.

**Nothing happens when I click Run.** Make sure both a camera dataset and a sensor dataset are
loaded (check the sidebar **Session data** panel).
        """
    )

st.divider()
st.caption("Project: https://github.com/cdavisv/Tidegate-Analysis-App · Distributed under the MIT License.")
