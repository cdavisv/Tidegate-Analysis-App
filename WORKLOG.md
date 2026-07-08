# Worklog — Tidegate Bird Patterns Analysis

Running log + working memory for agents. Newest on top. Keep concise.

## Working memory (essential, non-obvious)
- MOUNT GOTCHA: repo is a Windows mount inside the Linux sandbox. Editing an EXISTING tracked file via the Edit/Write tools updates the app view but bash/git can read STALE bytes (or the file gets trailing NUL padding / truncation when new content is shorter). NEW paths read fine. Reliable fix: rewrite bytes with `python3 -c "open(p,'wb').write(b)"` (this truncates), then re-read to confirm. Mount deletes often fail. Tracked files show as fully "modified" in git due to CRLF vs the LF-committed base — normalize to LF before `git add`.
- TESTING: deps at /tmp/pylibs (system pip fails; /sessions home full). Run `PYTHONPATH=/tmp/pylibs PYTHONPYCACHEPREFIX=/tmp/pyc python3 -m pytest`; clear /tmp/pyc after edits (stale .pyc).
- ARCHITECTURE: `app.py` (st.navigation) entry; pages in `views/`. Detection in `vision/` (demo | megadetector | openai behind Detector ABC; `camera_csv.py` emits the wide schema the loader expects). Weather in `weather/` (open-meteo | noaa | synoptic | csv). `pipeline_runner.run_full_analysis(camera, water)` wraps the dual-framework analysis. Session keys + theme in `ui_common.py`; config in `config.py`/`config.json`.
- CONTRACTS: camera CSV = 1 row/image; blank `Species 1` = valid no-detection record (dual-framework needs these). `data_loader.load_and_prepare_water_data` now keeps ALL numeric sensor columns so weather flows into the combined dataset.

## Log
### 2026-07-07 — Full pipeline shipped (agent: fable-5)
Delivered: image detection (demo / MegaDetector+SpeciesNet / OpenAI GPT vision) -> camera dataset; weather import (Open-Meteo / NOAA / Synoptic / CSV) with nearest-time merge; polished multi-page app + in-app Help; new Weather-Patterns analysis view; demo quick-start.
Bug fixes: removed hardcoded dataset numbers in comprehensive_analysis; UTF-8 BOM handling in data_loader; guarded Depth/Depth_Inside in data_combiner; tidal-quantile guard; water loader retains all numeric columns.
Tests: 33 pytest (vision 10, weather 21, pipeline 2) pass; AppTest boots + navigates all 5 pages + runs analysis end-to-end (37,017 periods, 406 detections on demo data). Verified from the mount (git's view).
New: app.py, ui_common.py, config.py/json, pipeline_runner.py, vision/, weather/, views/, tests/, .streamlit/, requirements-cv.txt, .env.example.

### 2026-07-07 — Review baseline
Reviewed core (main.py dual-framework, data_loader, data_combiner, comprehensive_analysis). Baseline PASSED on willanch CSVs (8,612 camera -> 37,017 combined, 15 species, 406 detections). Confirmed the camera CSV wide schema (Species N + image path cols + DateTime) that CV/LLM must emit.
