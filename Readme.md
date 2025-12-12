<!-- TODO: Complete Readme Template -->
<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->
<a id="readme-top"></a>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
<a id="readme-top"></a>

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![project_license][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<br />
<div align="center">
  <h3 align="center">Wildlife Detection and Tide Gate Analysis</h3>

  <p align="center">
    A data science pipeline for analyzing wildlife detection patterns in relation
    to tide dynamics, gate configurations, and environmental conditions using
    camera trap and sensor data.
    <br />
    <br />
    <a href="#getting-started"><strong>Get Started »</strong></a>
    &middot;
    <a href="https://github.com/your_username/your_repo/issues">Report Bug</a>
    &middot;
    <a href="https://github.com/your_username/your_repo/issues">Request Feature</a>
  </p>
</div>

---

## Table of Contents

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#analysis-framework">Analysis Framework</a></li>
    <li><a href="#built-with">Built With</a></li>
    <li><a href="#getting-started">Getting Started</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>

---

## About The Project

This project analyzes wildlife camera trap detections in tidal environments to
understand how animal activity and detection success vary with:

- Tide gate opening configurations
- Tidal flow states
- Environmental conditions
- Temporal patterns

A key goal of the project is to **separate operational bias from biological
behavior** by comparing two complementary analytical frameworks.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Analysis Framework

The project implements multiple complementary analytical layers to separate
operational bias from biological behavior.

### 1. Camera Activity Pattern Analysis
- Treats all monitoring periods as potential observation windows
- Measures when cameras were operational
- Identifies equipment and environmental biases in data collection

### 2. Wildlife Detection Efficiency Analysis
- Restricts analysis to periods when cameras were active
- Measures detection success when monitoring was occurring
- Focuses on animal behavior rather than equipment performance

### 3. Tidal Cycle and Phase Analysis
- Classifies tidal states (rising, falling, slack tides)
- Models continuous tidal phase across the full tidal cycle
- Identifies peak wildlife detection periods relative to tidal motion
- Analyzes species-specific tidal preferences

Together, these layers support robust ecological interpretation of wildlife
camera trap data in managed tidal systems.


<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Built With

- Python 3
- pandas
- NumPy
- SciPy
- statsmodels
- Plotly
- matplotlib

<p align="right">(<a href="#readme-top">back to top</a>)</p>

---

## Getting Started

### Prerequisites

- Python 3.9 or newer
- pip or conda

Install required packages:

```sh
pip install pandas numpy scipy statsmodels plotly matplotlib
```

## Installation

### Clone the repository:
```sh
git clone https://github.com/your_username/your_repo.git
```

### Navigate into the project directory:
```sh
cd your_repo
```

### Prepare your input CSV files (camera data and water/tide data)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Usage

Run the full analysis pipeline:
```sh
python main.py
```

The pipeline produces:

* A combined and interpolated camera + sensor dataset
* Species diversity and detection summaries
* Environmental and gate configuration detection analyses
* Tidal state and tidal phase detection analyses
* Species-specific tidal preference tables
* Interactive and static visualizations (HTML and PNG)
* All plots and tables are saved to disk for reproducibility and reporting.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Roadmap

Add mixed-effects models for repeated camera locations

Add spatial analysis support

Improve automated report generation

Add configuration file support

See the open issues for proposed features and known limitations.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contributing

Contributions are welcome, especially in the areas of:

Ecological modeling

Statistical validation

Visualization improvements

Performance optimization

Please fork the repository and submit a pull request.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## License

Distributed under the project license. See LICENSE for details.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

## Contact

Charles Davis
LinkedIn: [https://www.linkedin.com/in/charles-a-davis-v/]

Project Link: https://github.com/cdavisv/Tidegate-Analysis-App

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
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
[product-screenshot]: images/screenshot.png
<!-- Shields.io badges. You can a comprehensive list with many more badges at: https://github.com/inttter/md-badges -->
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 