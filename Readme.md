<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />
<div align="center">
  <h3 align="center">Wildlife Detection & Tide Gate Analysis</h3>

  <p align="center">
    An end-to-end Streamlit pipeline that turns raw camera-trap images into
    ecological insight — detecting species with computer vision or an LLM,
    importing weather &amp; tide data, and analyzing how wildlife activity tracks
    the tidal cycle, gate operations, and environmental conditions.
    <br />
    <br />
    <a href="#getting-started"><strong>Get Started</strong></a>
    &middot;
    <a href="https://github.com/cdavisv/Tidegate-Analysis-App/issues">Report Bug</a>
    &middot;
    <a href="https://github.com/cdavisv/Tidegate-Analysis-App/issues">Request Feature</a>
  </p>
</div>

---

## Table of Contents

- [About The Project](#about-the-project)
- [The Pipeline](#the-pipeline)
- [Analysis Framework](#analysis-framework)
- [Built With](#built-with)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Input Data Formats](#input-data-formats)
- [Usage](#usage)
- [Detectors](#detectors)
- [Weather &amp; Tide Sources](#weather--tide-sources)
- [Project Structure](#project-structure)
- [Outputs](#outputs)
- [Testing](#testing)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## About The Project

This project analyzes wildlife camera-trap detections in tidal environments to
understand how animal activity and detection success vary with:

- Tide gate opening configurations (MTR and top hinge gates)
- Tidal flow states (rising, falling, high slack, low slack) and continuous phase
- Environmental conditions (temperature, water depth, wind, humidity, pressure, precipitation)
- Temporal patterns (hourly, daily, seasonal)

It now covers the **whole workflow** — from a folder of trail-camera images to a
finished analysis — with a polished multi-page web app. A key analytical goal is
to **separate operational bias from biological behavior** by comparing two
complementary frameworks.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## The Pipeline

```
 Camera images ──▶  Image Detection  ──▶  Camera dataset (CSV)
 (folder/upload)    demo · MegaDetector/SpeciesNet · OpenAI GPT      │
                                                                     ▼
 Weather/Tide  ──▶  Weather Import   ──▶  Sensor + weather timeline ─┤
 Open-Meteo/NOAA/Synoptic/CSV                                        │
                                                                     ▼
                                         Analysis (dual-framework) ──▶ Insights
                                         species · gates · tides · weather
```

1. **Image Detection** — Identify wildlife in images and generate the wide-format
   *camera dataset*. Three interchangeable engines:
   - **Demo** — synthetic, deterministic detections (no models/keys) for testing.
   - **MegaDetector + SpeciesNet** — local computer vision (AddaxAI-style) via
     [Pytorch-Wildlife](https://github.com/microsoft/CameraTraps).
   - **OpenAI GPT vision** — a multimodal LLM identifies species directly.
2. **Weather Import** — Fetch weather &amp; tide data from Open-Meteo, NOAA, or
   Synoptic (or upload a CSV) and merge it onto the sensor timeline.
3. **Analysis** — Combine camera + sensor data and run the dual-framework
   analysis, exploring interactive results including a new **Weather Patterns** view.

Each step hands its output to the next through the app session; you can also start
at any step by uploading the relevant CSV.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Analysis Framework

### 1. Camera Activity Pattern Analysis
- Treats all monitoring periods as potential observation windows.
- **Metric:** Camera Activity Rate = Camera Active Periods / All Time Periods.
- Reveals equipment performance and operational bias.

### 2. Wildlife Detection Efficiency Analysis
- Restricts analysis to periods when cameras were active.
- **Metric:** Detection Rate = Animal Detections / Camera Observations.
- Reveals animal behavior and optimal monitoring conditions.

### 3. Tidal Cycle, Gate, and Weather Analysis
- Classifies tidal states and models a continuous tidal phase (0 = low, 0.5 = high).
- Hypothesis tests across MTR / top-hinge gate combinations.
- Detection rate across binned weather variables (temperature, wind, humidity, …).

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Built With

- [Python 3.9+](https://www.python.org/) · [Streamlit](https://streamlit.io/) (multi-page app)
- [pandas](https://pandas.pydata.org/) · [NumPy](https://numpy.org/) · [SciPy](https://scipy.org/) · [statsmodels](https://www.statsmodels.org/)
- [Plotly](https://plotly.com/python/) · [matplotlib](https://matplotlib.org/) · [seaborn](https://seaborn.pydata.org/)
- [Pillow](https://python-pillow.org/) · [requests](https://requests.readthedocs.io/) · [OpenAI](https://platform.openai.com/) (vision)
- Optional local CV: [Pytorch-Wildlife / MegaDetector](https://github.com/microsoft/CameraTraps), [SpeciesNet](https://github.com/google/cameratrapai)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Getting Started

### Prerequisites
- Python 3.9 or newer, and `pip`.

### Installation
```sh
git clone https://github.com/cdavisv/Tidegate-Analysis-App.git
cd Tidegate-Analysis-App
pip install -r requirements.txt
```

Optional — local computer-vision detection (MegaDetector + SpeciesNet):
```sh
pip install -r requirements-cv.txt      # heavy; a GPU is recommended
```

### Run the app
```sh
streamlit run app.py
```

> The legacy single-page app (`streamlit run main.py`) still works, but `app.py`
> is the new multi-page front door.

**Fastest tour:** Home → *Load Willanch demo data* → Analysis → *Run full analysis*.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Configuration

Non-secret defaults live in `config.json` (site name, coordinates, model names).
Secrets are read from the environment only and never written to disk. Copy
`.env.example` to `.env` (or export the variables):

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI GPT vision detector |
| `OPENAI_VISION_MODEL` | Override the default model (`gpt-5.5`) |
| `SYNOPTIC_TOKEN` | Synoptic/MesoWest weather |
| `TIDEGATE_LAT`, `TIDEGATE_LON`, `TIDEGATE_TZ`, `TIDEGATE_SITE` | Field-site defaults for weather/tide fetches |

The **Help & Docs** page in the app shows your current effective configuration.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Input Data Formats

### Camera dataset (CSV)
One row per image/observation. Requires a `DateTime` column (or `Date` + `Time`),
and `Species 1` (blank when nothing was detected), with optional `Species 2 …`,
`Species N Count`, and `Notes N`. Rows with a blank `Species 1` are preserved as
**no-animal camera records** — essential for the dual-framework analysis. The
Image Detection page produces exactly this shape automatically.

### Water / tide / sensor dataset (CSV)
Requires `DateTime` (or `Date` + `Time`). Recognized columns include
`Gate Opening MTR [Degrees]`, `Gate Opening Top Hinge [Degrees]`,
`Tidal Level Outside Tidegate [m]` (→ `Depth`), `Tidal Level Inside Tidegate [m]`,
`Air Temp [C]`, and `Wind Speed [km/h]`. **All additional numeric columns are now
retained** (humidity, pressure, precipitation, radiation, …) and flow into the
combined dataset for weather analysis. A UTF-8 BOM on the header is handled, and
zero depths are treated as sensor errors (→ NaN).

### Weather CSV (for upload)
Bracketed-unit headers such as `Air Temp [C]`, `Wind Speed [km/h]`,
`Precipitation [cm]`, `Barometric Pressure [mbar]`, `Humidity [%]` are
auto-detected and unit-converted to the normalized schema.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Usage

1. **Detect / import images** — On *Image Detection*, choose a detector, point it
   at a folder or upload images, and run. Download the generated camera CSV or
   pass it straight to the analysis.
2. **Import weather** — On *Weather Import*, pick a source (or upload a CSV),
   fetch, preview, and merge onto your sensor timeline.
3. **Analyze** — On *Analysis*, confirm the camera + sensor data, click *Run full
   analysis*, and explore results across the tabs. Download the combined dataset,
   per-analysis CSVs, and the console log.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Detectors

| Engine | Needs | Best for |
|--------|-------|----------|
| **Demo** | nothing | Trying the pipeline end-to-end without models/keys (synthetic data). |
| **MegaDetector + SpeciesNet** | `requirements-cv.txt` (PyTorch, Pytorch-Wildlife, optional SpeciesNet) | Local, private, high-volume camera-trap processing. |
| **OpenAI GPT vision** | `OPENAI_API_KEY` | Tricky frames where a multimodal model helps; no local GPU. |

The local CV path is inspired by and interoperable with the
[AddaxAI](https://github.com/PetervanLunteren/AddaxAI) ecosystem built around
MegaDetector and SpeciesNet. Capture timestamps are read from EXIF, then the
filename, then file modification time.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Weather & Tide Sources

All free, all normalized to a common schema (`Air_Temp_C`, `Wind_Speed_km_h`,
`Precipitation_cm`, `Barometric_Pressure_mbar`, `Humidity_pct`,
`Solar_Radiation_W_m2`, `Water_Level_m`):

- **Open-Meteo** *(recommended)* — no key; historical archive + forecast.
- **NOAA** — NWS station observations + CO-OPS tide water level (by station id).
- **Synoptic / MesoWest** — dense station network; free token.
- **CSV upload** — bring your own weather-station export.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Project Structure

```
Tidegate-Analysis-App/
├── app.py                         # Multi-page Streamlit entry (st.navigation)
├── views/                         # Page scripts
│   ├── home.py                    #   overview + demo quick-start
│   ├── detection.py               #   image detection (CV + LLM)
│   ├── weather_import.py          #   weather/tide fetch + merge
│   ├── analysis.py                #   dual-framework analysis + results
│   └── help.py                    #   in-app documentation
├── ui_common.py                   # Shared theme, header, session-state contract
├── config.py / config.json        # Site + model configuration
├── pipeline_runner.py             # Reusable analysis pipeline (frames or files)
├── vision/                        # Image detection subsystem
│   ├── schema.py                  #   detection dataclasses + species mapping
│   ├── camera_csv.py              #   detections → wide camera CSV
│   ├── image_source.py            #   folder/upload ingestion + EXIF time
│   ├── base.py                    #   Detector ABC (batch + progress)
│   ├── demo_detector.py           #   synthetic detector
│   ├── megadetector.py            #   MegaDetector + SpeciesNet (import-guarded)
│   ├── llm_openai.py              #   OpenAI GPT vision detector
│   └── pipeline.py                #   orchestration + registry
├── weather/                       # Weather import subsystem
│   ├── sources.py                 #   Open-Meteo / NOAA / Synoptic
│   └── normalize.py               #   schema, CSV import, merge
├── data_loader.py                 # Camera & water loading, species expansion
├── data_combiner.py               # Camera + sensor merge and interpolation
├── comprehensive_analysis.py      # Dual-framework analysis engine
├── species_analysis.py            # Species diversity metrics
├── environmental_analysis.py      # Environmental factor detection rates
├── bird_tide_analysis.py          # Wildlife-tide-gate interactions
├── gate_combination_analysis.py   # Multi-gate hypothesis testing
├── tide_cycle_analysis.py         # Tidal phase & species preferences
├── analysis.py                    # Chi-square / GLM statistics
├── visualization.py               # Core Plotly/matplotlib figures
├── additional_visualizations.py   # Method-comparison dashboards
├── main.py                        # Legacy single-page app (still works)
├── tests/                         # pytest suite (vision + weather + pipeline)
├── requirements.txt / requirements-cv.txt
└── output_plots/                  # Generated interactive HTML visualizations
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Outputs

The pipeline produces a combined dataset CSV, an analysis log, per-analysis CSV
downloads (species summary, gate interactions, tide preferences, environmental
rates), interactive Plotly HTML plots in `output_plots/`, and annotated tidal
hypothesis PNGs. A generated camera dataset CSV is also downloadable from the
Image Detection page.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Testing

```sh
pip install pytest
pytest -q            # unit tests for vision, weather, and the camera-CSV contract
```

The suite runs fully offline (network calls are mocked). An end-to-end smoke test
drives the demo detector through the analysis pipeline.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Roadmap

- [x] Image → dataset generation via computer vision (MegaDetector/SpeciesNet)
- [x] LLM (OpenAI GPT vision) image analysis option
- [x] Weather-station / tide data import (Open-Meteo, NOAA, Synoptic, CSV)
- [x] Multi-page app with in-app help and a weather-patterns view
- [x] Unit tests for detection, weather, and data loading
- [ ] Mixed-effects models for repeated camera locations
- [ ] Spatial analysis support
- [ ] Automated PDF/HTML report generation
- [ ] Package as an installable Python module

See the [open issues](https://github.com/cdavisv/Tidegate-Analysis-App/issues) for
proposed features and known limitations.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Contributing

Contributions are welcome — ecological modeling, statistical validation,
visualization, detector back-ends, and performance. Fork, branch
(`git checkout -b feature/your-feature`), commit, push, and open a Pull Request.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## License

Distributed under the MIT License. See [`License.md`](License.md) for details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Contact

Charles Davis - [LinkedIn](https://www.linkedin.com/in/charles-a-davis-v/)

Project Link: [https://github.com/cdavisv/Tidegate-Analysis-App](https://github.com/cdavisv/Tidegate-Analysis-App)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- MARKDOWN LINKS & IMAGES -->
[contributors-shield]