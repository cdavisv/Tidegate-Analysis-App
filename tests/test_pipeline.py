"""End-to-end smoke test: demo detector -> camera CSV -> full analysis."""

import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import pipeline_runner
from vision import get_detector, run_detection
from vision.image_source import ImageRef


def _synthetic_sensor(base, n):
    rng = np.random.default_rng(42)
    times = [base + timedelta(minutes=15 * i) for i in range(n)]
    return pd.DataFrame(
        {
            "DateTime": times,
            "Gate_Opening_MTR_Deg": rng.uniform(0, 80, n),
            "Gate_Opening_Top_Hinge_Deg": rng.uniform(0, 40, n),
            "Depth": rng.uniform(0.2, 3.0, n),
            "Air_Temp_C": rng.uniform(5, 20, n),
            "Humidity_pct": rng.uniform(40, 100, n),
        }
    )


def test_full_analysis_smoke():
    base = datetime(2022, 1, 1)
    n = 400
    sensor = _synthetic_sensor(base, n)

    refs = [ImageRef(path=f"/i/{i}.jpg", datetime=base + timedelta(minutes=15 * i)) for i in range(n)]
    camera_df, _ = run_detection(refs, get_detector("demo", detection_probability=0.4, seed=3))

    with tempfile.TemporaryDirectory() as t:
        res = pipeline_runner.run_full_analysis(
            camera_df, sensor,
            output_dir=os.path.join(t, "plots"),
            combined_csv_path=os.path.join(t, "combined.csv"),
        )

    for key in ["combined_df", "comprehensive", "species_summary", "env_results",
                "bird_tide_results", "figures"]:
        assert key in res

    comp = res["comprehensive"]["comparison"]
    assert comp["camera_periods"] > 0
    assert comp["total_periods"] >= comp["camera_periods"]
    # weather column retained through the loader and available for analysis
    assert "Humidity_pct" in res["combined_df"].columns
    assert isinstance(res["figures"], dict) and len(res["figures"]) > 0


def test_run_full_analysis_bad_input_raises():
    import pytest

    with pytest.raises(Exception):
        pipeline_runner.run_full_analysis(
            pd.DataFrame({"nope": [1, 2, 3]}),
            pd.DataFrame({"nope": [1, 2, 3]}),
        )
