# Worklog — Tidegate Bird Patterns Analysis

Running log + working memory for agents. Newest on top. Keep concise.

## Working memory (essential, non-obvious)
- MOUNT GOTCHA: repo is a Windows mount inside the Linux sandbox. Editing an EXISTING tracked file via the Edit/Write tools updates the app view but bash/git can read STALE bytes (or the file gets trailing NUL padding / truncation when new content is shorter). NEW paths read fine. Reliable fix: rewrite bytes with `python3 -c "open(p,'wb').write(b)"` (this truncates), then re-read to confirm. Mount deletes often fail. Tracked files show as fully "modified" in git due to CRLF vs the LF-committed base — normalize to LF before `git add`.
- TESTING: deps at /tmp/pylibs (system pip fails; /sessions home full). Run `PYTHONPATH=/tmp/pylibs PYTHONPYCACHEPREFIX=/tmp/pyc python3 -m pytest`; clear /tmp/pyc after edits (stale .pyc).
- ARCHITECTURE: `app.py` (st.navigation) entry; pages in `views/`. Detection in `vision/` (demo | megadetector | openai behind Detector ABC; `camera_csv.py` emits the wide schema the loader expects). Weather in `weather/` (open-meteo | noaa | synoptic | csv). `pipeline_runner.run_full_analysis(camera, water)` wraps the dual-framework analysis. Session keys + theme in `ui_common.py`; config in `config.py`/`config.json`.
- CONTRACTS: camera CSV = 1 row/image; blank `Species 1` = valid no-detection record (dual-framework needs these). `data_loader.load_and_prepare_water_data` now keeps ALL numeric sensor columns so weather flows into the combined dataset. All tidal quantile `pd.cut` sites are guarded against non-monotonic (degenerate/constant-Depth) edges. A camera dataset can be generated with NO image files via `vision.demo_data.synthetic_image_refs` (Detection → ✨ Demo set).

## Log

### 2026-07-09 — Verification + robustness pass (agent: opus / cowork)
Reviewed and ran the whole app headless — all 6 pages boot; full end-to-end via the
Home one-click demo = 37,017 periods / 8,612 camera / 406 detections / 16 figures;
live Open-Meteo fetch verified against the real API. Fixed a real crash: the tidal
`pd.cut` in `additional_visualizations.create_environmental_effectiveness_charts`
raised "bins must increase monotonically" when camera-active Depth was
(near-)constant, which aborted the ENTIRE analysis (that call was unguarded).
Applied the existing monotonic-edge guard there and to the two other unguarded
quantile cuts (`comprehensive_analysis`, `environmental_analysis`), and wrapped
`create_all_additional_visualizations` in `pipeline_runner` so no single bonus
visualization can abort a run. New: `vision/demo_data.synthetic_image_refs` + a
Detection-page "✨ Demo set (no files)" tab — a one-click image→dataset demo (the
demo detector never reads pixels; generated filenames carry timestamps spanning the
bundled sensor window, so results line up with the demo tide/weather data). Tests
56 → 63 (new `tests/test_demo_and_regression.py`: demo generator + tidal-guard
regressions). Prior hardening is committed as b09e727.

### 2026-07-09 — Pre-release hardening (agent: fable-5)
Repaired a stray `*kwargs)` at EOF of `weather/sources.py` (SyntaxError that killed
every page importing `weather`, i.e. the whole app). That was **working-tree-only**
corruption — HEAD was already clean — so the repair restored the file to match HEAD
and is not in this commit's diff (future agents: the mount can reintroduce this; a
repo-wide `py_compile` now guards it). Fixed (committed) `weather_import.py` calling
NOAA `fetch(..., tide_station=)` instead of `station_id=`.
Added `ui.page_link` safe wrapper (pages no longer crash when rendered outside
`st.navigation`) and routed all `st.page_link` calls through it. Home: one-click
**Load demo & run full analysis** (loads bundled CSVs + runs the pipeline in one
button, via new `ui.load_sample_data`). Analysis → Weather Patterns: added a
cross-variable point-biserial correlation table. `visualization.save_plot`: guard
optional `kaleido` (skip PNG export quietly instead of per-figure error spam) and
`groupby(observed=False)` to silence a pandas FutureWarning. Detection: preview
gallery of detected frames. Tests: new `tests/test_app_smoke.py` (module imports +
repo-wide py_compile + NOAA signature guard + AppTest boot of all pages). Suite
33 → 56 passing. Re-verified E2E on the real Willanch CSVs: 37,017 periods /
8,612 camera / 406 detections; all 8 analysis tabs render exception-free.

### 2026-07-07 — Full pipeline shipped (agent: fable-5)
Delivered: image detection (demo / MegaDetector+SpeciesNet / OpenAI GPT vision) -> camera dataset; weather import (Open-Meteo / NOAA / Synoptic / CSV) with nearest-time merge; polished multi-page app + in-app Help; new Weather-Patterns analysis view; demo quick-start.
Bug fixes: removed hardcoded dataset numbers in comprehensive_analysis; UTF-8 BOM handling in data_loader; guarded Depth/Depth_Inside in data_combiner; tidal-quantile guard; water loader retains all numeric columns.
Tests: 33 pytest (vision 10, weather 21, pipeline 2) pass; AppTest boots + navigates all 5 pages + runs analysis end-to-end (37,017 periods, 406 detections on demo data). Verified from the mount (git's view).
New: app.py, ui_common.py, config.py/json, pipeline_runner.py, vision/, weather/, views/, tests/, .streamlit/, requirements-cv.txt, .env.example.

### 2026-07-07 — Review baseline
Reviewed core (main.py dual-framework, data_loader, data_combiner, comprehensive_analysis). Baseline PASSED on willanch CSVs (8,612 camera -> 37,017 combined, 15 species, 406 detections). Confirmed the camera CSV wide schema (Species N + image path cols + DateTime) that CV/LLM must emit.
