<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />
<div align="center">
  <h3 align="center">Wildlife Detection and Tide Gate Analysis</h3>

  <p align="center">
    A Streamlit web application for analyzing wildlife camera trap detections
    in relation to tide dynamics, gate configurations, and environmental
    conditions using a dual-framework approach that separates operational bias
    from biological behavior.
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
- [Analysis Framework](#analysis-framework)
- [Built With](#built-with)
- [Getting Started](#getting-started)
- [Input Data Formats](#input-data-formats)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Outputs](#outputs)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## About The Project

This project analyzes wildlife camera trap detections in tidal environments to
understand how animal activity and detection success vary with:

- Tide gate opening configurations (MTR and top hinge gates)
- Tidal flow states (rising, falling, high slack, low slack)
- Environmental conditions (temperature, water depth, wind speed)
- Temporal patterns (hourly, daily, seasonal)

A key goal is to **separate operational bias from biological behavior** by
comparing two complementary analytical frameworks. The application combines
camera observation data with tidal/environmental sensor data, performs
statistical analysis (chi-square tests, GLM modeling), and generates
interactive Plotly visualizations.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Analysis Framework

The project implements multiple complementary analytical layers.

### 1. Camera Activity Pattern Analysis
- Treats all monitoring periods as potential observation windows
- Measures when cameras were operational relative to environmental conditions
- Identifies equipment and environmental biases in data collection
- **Metric:** Camera Activity Rate = Camera Active Periods / All Time Periods

### 2. Wildlife Detection Efficiency Analysis
- Restricts analysis to periods when cameras were active
- Measures detection success when monitoring was occurring
- Focuses on animal behavior rather than equipment performance
- **Metric:** Detection Rate = Animal Detections / Camera Observations

### 3. Tidal Cycle and Phase Analysis
- Classifies tidal states (rising, falling, high slack, low slack)
- Models continuous tidal phase across the full tidal cycle (0 = low tide, 0.5 = high tide, 1 = next low tide)
- Identifies peak wildlife detection periods relative to tidal motion
- Analyzes species-specific tidal preferences

Together, these layers support robust ecological interpretation of wildlife
camera trap data in managed tidal systems.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Built With

- [Python 3.9+](https://www.python.org/)
- [Streamlit](https://streamlit.io/) - Web application framework
- [pandas](https://pandas.pydata.org/) - Data manipulation and analysis
- [NumPy](https://numpy.org/) - Numerical computing
- [SciPy](https://scipy.org/) - Statistical analysis (chi-square tests, signal processing)
- [statsmodels](https://www.statsmodels.org/) - GLM modeling
- [Plotly](https://plotly.com/python/) - Interactive visualizations
- [matplotlib](https://matplotlib.org/) - Static visualizations
- [seaborn](https://seaborn.pydata.org/) - Statistical plot styling

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Getting Started

### Prerequisites

- Python 3.9 or newer
- pip (or conda)

### Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/cdavisv/Tidegate-Analysis-App.git
   ```

2. Navigate into the project directory:
   ```sh
   cd Tidegate-Analysis-App
   ```

3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Input Data Formats

The application requires two CSV files uploaded through the Streamlit interface.

### Camera Data CSV

Must contain a `DateTime` column (or separate `Date` and `Time` columns that will be combined automatically). Species observations are stored in wide format with repeating column groups:

| Column | Required | Description |
|--------|----------|-------------|
| `DateTime` | Yes* | Timestamp of observation |
| `Date` | Yes* | Date of observation (used if `DateTime` is absent) |
| `Time` | Yes* | Time of observation (used if `DateTime` is absent) |
| `Species 1` | Yes | Name of first species observed (blank if none detected) |
| `Species 1 Count` or `Species Count 1` | No | Count of individuals for Species 1 |
| `Notes 1` | No | Observation notes for Species 1 |
| `Species 2`, `Species 2 Count`, `Notes 2` | No | Additional species in the same observation |
| `Species N`, `Species N Count`, `Notes N` | No | Up to N species per observation row |

\*Either `DateTime` or both `Date` and `Time` must be present.

Rows where all species fields are empty are treated as "no animals detected" camera activity records. These records are critical for the dual-framework analysis, as they distinguish camera operational time from actual wildlife detections.

The application automatically standardizes common species names to scientific nomenclature (e.g., "canada goose" to "Branta canadensis").

### Water / Tide Sensor Data CSV

Must contain a `DateTime` column (or separate `Date` and `Time` columns). Columns are automatically renamed internally to standardized names.

| Column | Required | Internal Name | Description |
|--------|----------|---------------|-------------|
| `DateTime` | Yes* | `DateTime` | Timestamp of measurement |
| `Gate Opening MTR [Degrees]` | Recommended | `Gate_Opening_MTR_Deg` | MTR gate opening angle in degrees |
| `Gate Opening Top Hinge [Degrees]` | Recommended | `Gate_Opening_Top_Hinge_Deg` | Top hinge gate opening angle in degrees |
| `Tidal Level Outside Tidegate [m]` | Recommended | `Depth` | Water depth outside the tide gate (meters) |
| `Tidal Level Inside Tidegate [m]` | No | `Depth_Inside` | Water depth inside the tide gate (meters) |
| `Air Temp [C]` | No | `Air_Temp_C` | Air temperature in Celsius |
| `Wind Speed [km/h]` | No | `Wind_Speed_km_h` | Wind speed in km/h |

\*Either `DateTime` or both `Date` and `Time` must be present.

If your columns already use the internal naming convention (e.g., `Gate_Opening_MTR_Deg`), they will be used directly without renaming.

Zero values in depth columns are automatically replaced with NaN, as zero depth typically indicates sensor error rather than an actual measurement.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Usage

Launch the Streamlit application:

```sh
streamlit run main.py
```

The application will open in your web browser. The workflow is:

1. **Upload Data** - Upload your Camera CSV and Water/Tide CSV files using the file upload widgets
2. **Run Analysis** - Click the "Run Full Analysis" button to execute the full pipeline
3. **View Results** - Browse the analysis console output and interactive visualizations organized by category:
   - Species and detection overview
   - Environmental and gate effects
   - Wildlife behavior and gate interactions
   - Tidal cycle and phase analysis
   - Method comparisons and performance dashboards
4. **Download Outputs** - Download the combined dataset CSV and the analysis log

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Project Structure

```
Tidegate-Analysis-App/
├── main.py                        # Streamlit app entry point and pipeline orchestration
├── data_loader.py                 # Camera and water data loading, species expansion
├── data_combiner.py               # Camera + sensor data merging and interpolation
├── comprehensive_analysis.py      # Dual-framework analysis engine
├── species_analysis.py            # Species diversity metrics and preferences
├── environmental_analysis.py      # Environmental factor detection rate analysis
├── bird_tide_analysis.py          # Wildlife-tide-gate interaction analysis
├── gate_combination_analysis.py   # Multi-gate combination hypothesis testing
├── tide_cycle_analysis.py         # Tidal cycle phase and species preference analysis
├── analysis.py                    # Statistical analysis: chi-square, GLM modeling
├── visualization.py               # Core Plotly/matplotlib visualization generation
├── additional_visualizations.py   # Method comparison dashboards and advanced charts
├── fieldinsertion.py              # CLI utility for CSV field correction
├── requirements.txt               # Python dependencies
├── License.md                     # MIT License
├── Readme.md                      # This file
└── output_plots/                  # Generated interactive HTML visualizations
```

### Data Flow

```
Camera CSV + Water/Tide CSV
        │
  data_loader.py  ─── Load, normalize, expand multi-species rows
        │
  data_combiner.py  ─── Merge datasets, interpolate water variables (15-min window)
        │
  ┌─────┼─────────────────────────────────────────────────┐
  │     │                                                 │
  │  comprehensive_analysis.py                            │
  │     ├── Camera Activity Pattern Analysis              │
  │     └── Wildlife Detection Efficiency Analysis        │
  │                                                       │
  │  species_analysis.py ── Species diversity metrics     │
  │  environmental_analysis.py ── Gate/tide/temp effects  │
  │  bird_tide_analysis.py ── Wildlife-tide interactions  │
  │  gate_combination_analysis.py ── Multi-gate hypotheses│
  │  tide_cycle_analysis.py ── Tidal phase preferences    │
  │  analysis.py ── Chi-square tests, GLM                 │
  └───────────────────────────────────────────────────────┘
        │
  visualization.py + additional_visualizations.py
        │
  output_plots/ (HTML) + combined_data_output.csv
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Outputs

The pipeline produces the following outputs:

| Output | Format | Description |
|--------|--------|-------------|
| Combined dataset | CSV | Camera observations merged with interpolated sensor data |
| Analysis log | TXT | Full console output from all analysis stages |
| Species summary | HTML | Top species by count and detection events |
| Gate detection rates | HTML | Detection rates by MTR and top hinge gate positions |
| Tidal level effects | HTML | Detection rates across tidal depth levels |
| Water parameter time series | HTML | Time series of available water quality parameters |
| Wildlife detection heatmap | HTML | Detection rates by gate status and tidal flow |
| Wildlife detection scatter | HTML | Detections vs gate angle and tidal change rate |
| Tidal state bar chart | HTML | Detection rates by rising/falling/slack tide |
| Tidal phase polar chart | HTML | Detection rates around the full tidal cycle |
| Species tide preferences | HTML | Heatmap of species-specific tidal state preferences |
| Method comparison | HTML | Side-by-side Camera Activity vs Detection Efficiency |
| Performance dashboard | HTML | Camera system performance gauges and data quality metrics |
| Hypothesis visualizations | PNG | Annotated tidal cycle diagrams with peak activity |

All interactive HTML plots are saved to the `output_plots/` directory. The combined dataset and analysis log are available for download directly from the Streamlit interface.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Roadmap

- [ ] Add mixed-effects models for repeated camera locations
- [ ] Add spatial analysis support
- [ ] Improve automated report generation
- [ ] Add configuration file support
- [ ] Add unit tests for data loading and analysis modules
- [ ] Package as installable Python module

See the [open issues](https://github.com/cdavisv/Tidegate-Analysis-App/issues) for proposed features and known limitations.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Contributing

Contributions are welcome, especially in the areas of:

- Ecological modeling
- Statistical validation
- Visualization improvements
- Performance optimization

To contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

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
[contributors-shield]: https://img.shields.io/github/contributors/cdavisv/Tidegate-Analysis-App.svg?style=for-the-badge
[contributors-url]: https://github.com/cdavisv/Tidegate-Analysis-App/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/cdavisv/Tidegate-Analysis-App.svg?style=for-the-badge
[forks-url]: https://github.com/cdavisv/Tidegate-Analysis-App/network/members
[stars-shield]: https://img.shields.io/github/stars/cdavisv/Tidegate-Analysis-App.svg?style=for-the-badge
[stars-url]: https://github.com/cdavisv/Tidegate-Analysis-App/stargazers
[issues-shield]: https://img.shields.io/github/issues/cdavisv/Tidegate-Analysis-App.svg?style=for-the-badge
[issues-url]: https://github.com/cdavisv/Tidegate-Analysis-App/issues
[license-shield]: https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge
[license-url]: https://opensource.org/licenses/MIT
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/charles-a-davis-v/
