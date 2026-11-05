import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from streamlit.testing.v1 import AppTest

from shot_predict.court import Court, ZONES
from shot_predict.dataset import ShotDataset
from shot_predict.model import ModelConfig, ShotModel
from shot_predict.predict import load_model, parser as export_parser, run as export
from shot_predict.storage import read_project, write_project
from shot_predict.train import parser as train_parser, run as train
from shot_predict.video import metadata, prepare_frame, read_frame


def test_video_dataset_shapes_and_pixels(synthetic_project):
    _, project = synthetic_project
    config = ModelConfig(image_size=32, samples=2, seconds=0.1, hidden=8)
    data = ShotDataset(project, project["shots"], config, [100, 200])
    assert len(data) == 8
    tensor, label = data[0]
    assert tensor.shape == (2, 3, 32, 32)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()
    assert label == ZONES.index("ad_deep")
    for example in data.examples:
        assert max(example["indices"]) < example["shot"]["contact_frame"]
    image = read_frame(project["videos"]["training"]["path"], 0)
    np.testing.assert_allclose(
        prepare_frame(image, 32, "far"),
        prepare_frame(cv2.rotate(image, cv2.ROTATE_180), 32, "near"),
    )


def test_model_shape_and_frozen_batchnorm():
    torch.set_num_threads(1)
    model = ShotModel(ModelConfig(image_size=32, samples=2, hidden=8, freeze_backbone=True))
    model.train()
    assert not model.backbone.training
    assert not any(p.requires_grad for p in model.backbone.parameters())
    output = model(torch.zeros(2, 2, 3, 32, 32))
    assert output.shape == (2, 6)
    torch.testing.assert_close(output.softmax(1).sum(1), torch.ones(2))


def test_training_and_export_end_to_end(synthetic_project, tmp_path):
    project_path, project = synthetic_project
    run_dir = tmp_path / "run"
    args = train_parser().parse_args([
        "--project", str(project_path), "--output", str(run_dir),
        "--epochs", "1", "--batch-size", "2", "--image-size", "32",
        "--samples", "2", "--seconds", "0.1", "--hidden", "8",
        "--lead-ms", "100", "200", "--device", "cpu", "--threads", "1",
        "--validation-match", "validation-match",
    ])
    with pytest.warns(UserWarning, match="Missing training zones"):
        checkpoint = train(args)
    assert checkpoint.is_file()
    report = json.loads((run_dir / "metrics.json").read_text())
    assert report["validation_matches"] == ["validation-match"]
    assert report["player_overlap"] == []
    assert set(report["train_shot_ids"]).isdisjoint(report["validation_shot_ids"])
    for metric in report["history"][0]["by_lead_ms"].values():
        assert 0 <= metric["top1"] <= 1
        assert metric["top2"] >= metric["top1"]
        assert np.array(metric["confusion_matrix"]).shape == (6, 6)
        assert sum(map(sum, metric["confusion_matrix"])) == 2
        assert set(metric["by_group"]) == {
            "hitter:near", "hitter:far", "shot_type:groundstroke", "shot_type:serve"
        }
    model, config, leads = load_model(checkpoint, torch.device("cpu"))
    assert config.samples == 2 and leads == [100, 200]
    assert not model.training
    project["shots"] = []
    write_project(project_path, project)
    output = tmp_path / "prediction.mp4"
    export_args = export_parser().parse_args([
        "--project", str(project_path), "--video-id", "validation",
        "--checkpoint", str(checkpoint), "--output", str(output),
        "--stride", "4", "--threshold", "0", "--device", "cpu", "--threads", "1",
    ])
    output_path, csv_path = export(export_args)
    info = metadata(output_path)
    assert info["frames"] == 80
    assert (info["width"], info["height"], info["fps"]) == (160, 120, 20)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and {r["hitter"] for r in rows} == {"near", "far"}
    for row in rows:
        values = [float(row[zone]) for zone in ZONES]
        assert len(values) == 6
        assert all(0 <= value <= 1 for value in values)
        assert sum(values) == pytest.approx(1, abs=1e-6)
        assert int(row["frame"]) >= 2
    assert not list(tmp_path.glob("*.tmp.*"))
    with pytest.raises(ValueError, match="already exists"):
        export(export_args)
    with pytest.raises(ValueError, match="already exists"):
        train(args)


def test_app_starts_empty(monkeypatch):
    import shot_predict.storage as storage

    monkeypatch.setattr(storage, "read_project", lambda path: {"version": 1, "videos": {}, "shots": []})
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30)
    app.run(timeout=30)
    assert not app.exception
    assert app.title[0].value == "Tennis shot lab"


def test_app_persists_workspace_actions_and_ignores_duplicates(synthetic_project, monkeypatch):
    from shot_predict import labeler

    project_path, project = synthetic_project
    monkeypatch.setenv("SHOT_LAB_PROJECT", str(project_path))
    monkeypatch.setattr(labeler, "video_url", lambda path: "/media/test.mp4")
    pending = []
    rendered = []

    def workspace(**kwargs):
        rendered.append(kwargs)
        return pending.pop(0) if pending else None

    monkeypatch.setattr(labeler, "workspace", workspace)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=30)
    app.run()
    assert not app.exception and rendered[-1]["video_id"] == "training"
    assert rendered[-1]["shot_count"] == 2
    court = Court(project["videos"]["training"]["corners"])
    action = {
        "event_id": "save-1", "action": "save_shot", "video_id": "training",
        "contact_frame": 31, "bounce_frame": 45, "hitter": "near",
        "shot_type": "groundstroke", "landing_pixel": list(court.target("ad_deep", "near")),
    }
    pending.append(action)
    app.run()
    assert not app.exception
    saved = read_project(project_path)
    assert len(saved["shots"]) == len(project["shots"]) + 1
    assert saved["shots"][-1]["contact_frame"] == 31
    assert rendered[-1]["acknowledgement"]["ok"]
    pending.append(action)
    app.run()
    assert len(read_project(project_path)["shots"]) == len(saved["shots"])
    pending.append({**action, "event_id": "bad-1", "bounce_frame": 20})
    app.run()
    assert not rendered[-1]["acknowledgement"]["ok"]
    assert len(read_project(project_path)["shots"]) == len(saved["shots"])
    pending.append({
        "event_id": "select-1", "action": "select_video", "video_id": "training",
        "selected_video": "validation",
    })
    app.run()
    assert not app.exception and rendered[-1]["video_id"] == "validation"
