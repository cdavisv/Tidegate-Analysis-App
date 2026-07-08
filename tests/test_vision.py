"""Unit tests for the vision detection subsystem (offline, no models/keys)."""

import os
import tempfile
from datetime import datetime

import pandas as pd
import pytest

from vision.schema import (
    to_scientific_name,
    DetectionItem,
    ImageDetection,
    CATEGORY_PERSON,
)
from vision.camera_csv import detections_to_camera_df, detections_to_long_df
from vision.demo_detector import DemoDetector
from vision.image_source import ImageRef
from vision.pipeline import get_detector, run_detection, summarize_detections
from vision.megadetector import MegaDetectorDetector, parse_megadetector_result
from vision.llm_openai import parse_llm_json, OpenAIVisionDetector, DEFAULT_VISION_MODEL


def test_scientific_mapping():
    assert to_scientific_name("Great Blue Heron") == "Ardea herodias"
    assert to_scientific_name("canada goose") == "Branta canadensis"
    assert to_scientific_name("Wombat") == "Wombat"  # title-cased fallback
    assert to_scientific_name(None) == ""
    assert to_scientific_name("   ") == ""


def test_detection_item_defaults():
    it = DetectionItem(label="Mallard")
    assert it.scientific_name == "Anas platyrhynchos"
    assert it.count == 1
    assert DetectionItem(label="X", count=0).count == 1  # clamped to >= 1


def test_image_detection_helpers():
    d = ImageDetection(
        image_path="a.jpg",
        items=[
            DetectionItem(label="Mallard", count=2),
            DetectionItem(category=CATEGORY_PERSON, label="Human"),
        ],
    )
    assert d.has_animal
    assert d.total_count == 2
    assert d.top_species() == "Mallard"
    assert len(d.animals) == 1  # person excluded


def test_camera_csv_empty_and_multispecies():
    dets = [
        ImageDetection(image_path="/x/empty.jpg", datetime=datetime(2022, 1, 1, 8, 0), items=[]),
        ImageDetection(
            image_path="/x/multi.jpg", datetime=datetime(2022, 1, 1, 9, 0),
            items=[
                DetectionItem(label="Mallard", count=2, confidence=0.8),
                DetectionItem(label="Great Blue Heron", count=1, confidence=0.9),
            ],
        ),
    ]
    df = detections_to_camera_df(dets)
    assert list(df.columns)[:5] == [
        "relative_path", "Full Path", "Corrected", "absolute_path", "data_type",
    ]
    assert "Species 1" in df.columns and "Species 2" in df.columns
    # empty image -> blank Species 1 (loader treats as no-detection)
    val = df.iloc[0]["Species 1"]
    assert val in ("", None) or pd.isna(val)
    # dominant species (higher count) lands in Species 1
    assert df.iloc[1]["Species 1"] == "Mallard"
    assert int(df.iloc[1]["Species 1 Count"]) == 2


def test_camera_csv_round_trips_through_loader():
    import data_loader

    dets = [
        ImageDetection(
            image_path=f"/x/{i}.jpg", datetime=datetime(2022, 1, 1, 8, i % 59),
            items=([DetectionItem(label="Mallard", count=1)] if i % 2 == 0 else []),
        )
        for i in range(20)
    ]
    df = detections_to_camera_df(dets)
    with tempfile.TemporaryDirectory() as t:
        p = os.path.join(t, "cam.csv")
        df.to_csv(p, index=False)
        loaded = data_loader.load_and_prepare_camera_data(p)
    assert loaded is not None
    assert len(loaded) >= 20  # detections + no-detection records


def test_long_df_shape():
    dets = [
        ImageDetection(image_path="/x/a.jpg", items=[DetectionItem(label="Mallard", count=2)]),
        ImageDetection(image_path="/x/b.jpg", items=[]),
    ]
    long = detections_to_long_df(dets)
    assert len(long) == 2
    assert set(["image", "species", "count", "detector"]).issubset(long.columns)


def test_demo_detector_deterministic():
    r = ImageRef(path="/img/frame.jpg")
    a = DemoDetector(seed=5).detect_image(r)
    b = DemoDetector(seed=5).detect_image(r)
    assert [(i.label, i.count) for i in a.items] == [(i.label, i.count) for i in b.items]


def test_pipeline_run_and_summary():
    refs = [ImageRef(path=f"/i/{i}.jpg", datetime=datetime(2022, 1, 1)) for i in range(60)]
    det = get_detector("demo", detection_probability=0.5, seed=1)
    df, dets = run_detection(refs, det)
    assert len(df) == 60
    s = summarize_detections(dets)
    assert s["n_images"] == 60
    assert s["n_with_animals"] >= 1
    assert 0 <= s["detection_rate"] <= 100


def test_megadetector_graceful_and_parse():
    md = MegaDetectorDetector()
    ok, reason = md.available()
    assert isinstance(ok, bool) and isinstance(reason, str)

    class FakeDets:
        xyxy = [[0, 0, 10, 10], [5, 5, 20, 20]]
        confidence = [0.9, 0.1]
        class_id = [0, 1]

    items = parse_megadetector_result({"detections": FakeDets()}, threshold=0.2)
    assert len(items) == 1  # low-confidence box filtered out
    assert items[0].category == "animal"


def test_llm_parse_and_unavailable():
    items = parse_llm_json(
        '```json\n{"detections":[{"common_name":"Mallard","count":3,"category":"animal"}]}\n```'
    )
    assert items[0].label == "Mallard"
    assert items[0].scientific_name == "Anas platyrhynchos"
    assert items[0].count == 3
    assert parse_llm_json("not json") == []
    oa = OpenAIVisionDetector(api_key=None)
    assert oa.available()[0] is False
    assert isinstance(DEFAULT_VISION_MODEL, str) and DEFAULT_VISION_MODEL
