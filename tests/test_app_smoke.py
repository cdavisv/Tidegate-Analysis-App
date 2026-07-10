"""App-level smoke tests.

These guard against the class of breakage where a single file has a syntax
error (which silently breaks the whole Streamlit app) or a page calls a function
with the wrong signature. They complement the focused unit tests in
``test_vision``/``test_weather``/``test_pipeline``.
"""

from __future__ import annotations

import glob
import importlib
import inspect
import os
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


# ---------------------------------------------------------------------------
# Every core module must import cleanly (catches syntax errors like a stray
# token at end-of-file, which otherwise only surfaces at runtime).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "module",
    [
        "config",
        "ui_common",
        "pipeline_runner",
        "weather",
        "weather.sources",
        "weather.normalize",
        "vision",
        "vision.pipeline",
        "vision.megadetector",
        "vision.llm_openai",
        "vision.demo_detector",
        "data_loader",
        "data_combiner",
        "comprehensive_analysis",
    ],
)
def test_core_module_imports(module):
    importlib.import_module(module)


def test_all_python_files_compile():
    """Byte-compile every .py file in the repo (guards against stray bytes)."""
    import py_compile

    checked = 0
    for path in glob.glob(os.path.join(REPO, "**", "*.py"), recursive=True):
        if "__pycache__" in path:
            continue
        py_compile.compile(path, doraise=True)
        checked += 1
    assert checked > 20


def test_noaa_fetch_accepts_station_id():
    """The weather page passes ``station_id=`` to NOAASource.fetch()."""
    from weather import NOAASource

    params = inspect.signature(NOAASource.fetch).parameters
    assert "station_id" in params
    assert "tide_station" not in params


def test_weather_sources_registry():
    import weather

    assert set(weather.SOURCES) == {"open-meteo", "noaa", "synoptic"}


# ---------------------------------------------------------------------------
# Every page must boot headless without raising (via the safe page_link wrapper
# the pages render even outside st.navigation).
# ---------------------------------------------------------------------------
PAGES = [
    "app.py",
    "views/home.py",
    "views/detection.py",
    "views/weather_import.py",
    "views/analysis.py",
    "views/help.py",
]


@pytest.mark.parametrize("page", PAGES)
def test_page_boots_without_exceptions(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(os.path.join(REPO, page), default_timeout=90)
    at.run()
    assert not at.exception, [str(e.value) for e in at.exception]
