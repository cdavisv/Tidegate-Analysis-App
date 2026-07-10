"""Tests for the synthetic demo-set generator and the tidal-binning regression.

Covers two additions made during pre-release hardening:

* ``vision.demo_data.synthetic_image_refs`` -- fabricates timestamped image
  references so the image -> dataset step is demoable without real files.
* The degenerate-quantile guard on the tidal ``pd.cut`` calls that previously
  raised ``ValueError: bins must increase monotonically`` when camera-active
  depth values were (near-)constant, aborting the whole analysis.
"""

from __future__ import annotations

import os
import sys
from datetime import date

import pandas as pd
import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)


# ---------------------------------------------------------------------------
# Synthetic demo image references
# ---------------------------------------------------------------------------
def test_synthetic_refs_count_sorted_and_windowed():
    from vision.demo_data import synthetic_image_refs, DEMO_START, DEMO_END

    refs = synthetic_image_refs(50, seed=3)
    assert len(refs) == 50
    times = [r.datetime for r in refs]
    assert times == sorted(times), "refs must be time-sorted"
    assert all(DEMO_START <= t <= DEMO_END for t in times), "within default window"
    assert all(r.datetime_source == "filename" for r in refs)


def test_synthetic_refs_are_reproducible():
    from vision.demo_data import synthetic_image_refs

    a = synthetic_image_refs(30, seed=7)
    b = synthetic_image_refs(30, seed=7)
    assert [r.path for r in a] == [r.path for r in b]


def test_synthetic_refs_filename_timestamp_roundtrips():
    """A generated filename must parse back to the same capture time."""
    from vision.demo_data import synthetic_image_refs
    from vision.image_source import read_capture_datetime

    r = synthetic_image_refs(5, seed=9)[0]
    dt, src = read_capture_datetime(r.path)
    assert src == "filename"
    assert dt == r.datetime


def test_synthetic_refs_custom_window_accepts_dates():
    from vision.demo_data import synthetic_image_refs

    refs = synthetic_image_refs(20, start=date(2023, 1, 1), end=date(2023, 1, 5), seed=1)
    lo, hi = pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-05 23:59:59")
    assert all(lo <= pd.Timestamp(r.datetime) <= hi for r in refs)


def test_demo_detector_over_synthetic_refs_builds_camera_dataset():
    """image -> dataset: demo detector on synthetic refs yields the wide schema."""
    from vision import get_detector, detections_to_camera_df
    from vision.demo_data import synthetic_image_refs

    refs = synthetic_image_refs(80, seed=5)
    dets = get_detector("demo", detection_probability=0.30, seed=5).detect_batch(refs)
    assert len(dets) == 80

    cam = detections_to_camera_df(dets)
    assert len(cam) == 80
    assert "Species 1" in cam.columns and "DateTime" in cam.columns

    blank = cam["Species 1"].isna() | (cam["Species 1"].astype(str).str.strip() == "")
    # The dual-framework analysis needs both detections and no-animal records.
    assert blank.any(), "expected some no-animal (blank Species 1) records"
    assert (~blank).any(), "expected some animal detections"


# ---------------------------------------------------------------------------
# Regression: degenerate tidal quantile bins must not crash the analysis
# ---------------------------------------------------------------------------
def _degenerate_camera_frame(n: int = 300) -> pd.DataFrame:
    """A combined-style frame whose camera-active rows have constant Depth.

    Constant depth makes the 25th and 75th percentiles equal, which produced
    non-monotonic ``pd.cut`` bin edges before the guard was added.
    """
    dt = pd.date_range("2022-01-01", periods=n, freq="h")
    half = n // 2
    return pd.DataFrame(
        {
            "DateTime": dt,
            "has_camera_data": [True] * half + [False] * (n - half),
            "Depth": [1.5] * n,  # constant -> degenerate quantiles
            "Gate_Opening_MTR_Deg": np.tile([0, 10, 45, 70], n // 4 + 1)[:n],
            "Air_Temp_C": np.linspace(5.0, 15.0, n),
        }
    )


def test_effectiveness_charts_survive_constant_depth():
    import additional_visualizations as av

    fig = av.create_environmental_effectiveness_charts(_degenerate_camera_frame())
    # Previously raised ValueError("bins must increase monotonically").
    assert fig is not None


def test_comprehensive_tidal_binning_survives_constant_depth():
    """The camera-activity tidal analysis must not raise on constant depth."""
    import comprehensive_analysis as ca

    df = _degenerate_camera_frame()
    # This exercises the combined_df tidal quantile cut (guarded).
    result = ca.analyze_environmental_factors_camera_activity(df.copy())
    assert isinstance(result, tuple) and len(result) == 4
