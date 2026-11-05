import json
import random

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from shot_predict.court import Court, ZONES
from shot_predict.dataset import augment_clip, split_exploratory_videos, split_matches, split_videos
from shot_predict.model import ModelConfig, ShotModel
from shot_predict.predict import load_model
from shot_predict.storage import add_shot, write_project
from shot_predict.train import cache_features, inverse_sqrt_class_weights, parser, run
from shot_predict.video import metadata


def test_exploratory_split_holds_out_whole_source_clips(tmp_path):
    project = {
        "videos": {
            key: {"match_id": "one-match", "path": str(tmp_path / f"{key}.mp4")}
            for key in ("first", "last")
        },
        "shots": [
            {"id": str(i), "video_id": key}
            for i, key in enumerate(("first", "last", "first", "last"))
        ],
    }
    train, validation, held = split_exploratory_videos(project, ["last"])
    assert held == ["last"]
    assert {s["video_id"] for s in train} == {"first"}
    assert {s["video_id"] for s in validation} == {"last"}
    with pytest.raises(ValueError, match="TWO"):
        split_matches(project, 42)
    with pytest.raises(ValueError, match="labeled video"):
        split_exploratory_videos(project, ["missing"])
    with pytest.raises(ValueError, match="leave at least one"):
        split_exploratory_videos(project, ["first", "last"])
    project["videos"]["last"]["path"] = project["videos"]["first"]["path"]
    with pytest.raises(ValueError, match="same source"):
        split_exploratory_videos(project, ["last"])


def test_clip_holdout_works_across_matches_and_never_splits_a_clip(tmp_path):
    project = {
        "videos": {
            "a1": {"match_id": "match-a", "path": str(tmp_path / "a1.mp4")},
            "a2": {"match_id": "match-a", "path": str(tmp_path / "a2.mp4")},
            "b1": {"match_id": "match-b", "path": str(tmp_path / "b1.mp4")},
            "b2": {"match_id": "match-b", "path": str(tmp_path / "b2.mp4")},
        },
        "shots": [{"id": str(i), "video_id": key} for i, key in enumerate(("a1", "a2", "b1", "b2", "a1", "b2", "a2"))],
    }
    train, validation, held = split_videos(project, ["a2", "b2"])
    assert held == ["a2", "b2"]
    assert {s["video_id"] for s in train} == {"a1", "b1"} and {s["video_id"] for s in validation} == {"a2", "b2"}
    assert len(train) + len(validation) == len(project["shots"])
    with pytest.raises(ValueError, match="exactly one labeled match"):
        split_exploratory_videos(project, ["a2"])  # the exploratory split stays single-match
    with pytest.raises(ValueError, match="labeled video"):
        split_videos(project, ["nope"])
    with pytest.raises(ValueError, match="leave at least one"):
        split_videos(project, ["a1", "a2", "b1", "b2"])


@pytest.mark.parametrize("zone", ZONES)
def test_mirrors_swap_ad_deuce_but_not_depth_or_middle(monkeypatch, zone):
    monkeypatch.setattr(random, "uniform", lambda low, high: 0 if low < 0 else 1)
    monkeypatch.setattr(random, "random", lambda: 0)
    raw = torch.linspace(0.1, 0.9, 32).view(1, 1, 1, 32).expand(2, 3, 32, 32)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    clips = (raw - mean) / std
    transformed, label = augment_clip(clips, ZONES.index(zone))
    lateral, depth = zone.split("_")
    expected = {"ad": "deuce", "deuce": "ad", "middle": "middle"}[lateral]
    assert ZONES[label] == f"{expected}_{depth}"
    torch.testing.assert_close(transformed, clips.flip(-1), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(transformed[0], transformed[1])


def test_augmentation_is_temporally_consistent_and_seeded():
    clips = torch.randn(1, 3, 32, 32).expand(4, -1, -1, -1).clone()
    random.seed(17)
    first, label = augment_clip(clips, 0)
    random.seed(17)
    second, second_label = augment_clip(clips, 0)
    torch.testing.assert_close(first, second)
    assert label == second_label
    assert torch.isfinite(first).all()
    assert not torch.equal(first, clips)
    for frame in first[1:]:
        torch.testing.assert_close(frame, first[0])


def test_frozen_feature_cache_preserves_forward_and_supports_backprop():
    torch.set_num_threads(1)
    config = ModelConfig(image_size=32, samples=2, hidden=8, freeze_backbone=True)
    model = ShotModel(config)

    class Clips(Dataset):
        examples = [{"label": 0}, {"label": 2}]
        images = torch.randn(2, 2, 3, 32, 32)

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return self.images[index], self.examples[index]["label"]

    data = Clips()
    cached = cache_features(model, data, torch.device("cpu"), 2, 1)
    assert len(cached) == 4
    assert len(cached.examples) == 4
    model.eval()
    torch.testing.assert_close(model(data.images), model.classify(cached.features[:2]))
    model.train()
    loss = model.classify(cached.features).sum()
    loss.backward()
    assert model.head.weight.grad is not None
    assert all(p.grad is None for p in model.backbone.parameters())


def test_inverse_sqrt_class_weights_use_present_classes():
    weights = inverse_sqrt_class_weights(np.array([100, 25, 4, 0, 9, 16]))
    assert weights[3] == 0
    assert weights[2] / weights[0] == pytest.approx(5.0)
    assert weights[[0, 1, 2, 4, 5]].mean() == pytest.approx(1.0)
    for invalid in (
        np.zeros(6), np.array([1, 2, 3]), np.array([1, 1, -1, 1, 1, 1]),
        np.array([1, 1, np.nan, 1, 1, 1]),
    ):
        with pytest.raises(ValueError):
            inverse_sqrt_class_weights(invalid)


@pytest.mark.parametrize("class_weighting", ["none", "inverse-sqrt"])
def test_exploratory_augmented_training_checkpoint(tmp_path, monkeypatch, class_weighting):
    import shot_predict.train as training

    torch.set_num_threads(1)
    court = Court([[20, 110], [140, 110], [140, 10], [20, 10]])
    project = {"version": 1, "videos": {}, "shots": []}
    for key in ("first", "last"):
        path = tmp_path / f"{key}.avi"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (160, 120))
        assert writer.isOpened()
        for i in range(80):
            writer.write(np.full((120, 160, 3), i * 3, np.uint8))
        writer.release()
        project["videos"][key] = {
            "path": str(path), "match_id": "same-match", "corners": court.corners,
            "near_player": "near", "far_player": "far", **metadata(path),
        }
        add_shot(project, key, 30, 42, "near", "groundstroke", court.target("ad_deep", "near"))
        add_shot(project, key, 50, 65, "far", "serve", court.target("deuce_short", "far"))
    project_path = tmp_path / "project.json"
    write_project(project_path, project)
    monkeypatch.setattr(training, "ShotModel", lambda config, pretrained: ShotModel(config))
    args = parser().parse_args([
        "--project", str(project_path), "--output", str(tmp_path / "run"),
        "--epochs", "1", "--batch-size", "2", "--image-size", "32",
        "--samples", "2", "--seconds", "0.1", "--hidden", "8",
        "--lead-ms", "100", "--threads", "1", "--pretrained", "--freeze-backbone",
        "--cache-features", "--augmentation-copies", "1",
        "--class-weighting", class_weighting,
        "--exploratory-validation-video", "last",
    ])
    with pytest.warns(UserWarning) as warnings:
        checkpoint = run(args)
    assert any("EXPLORATORY" in str(w.message) for w in warnings)
    report = json.loads((args.output / "metrics.json").read_text())
    assert report["evaluation_scope"] == "exploratory_same_match_video_holdout"
    assert report["validation_matches"] == []
    assert report["validation_video_ids"] == ["last"]
    assert report["training_examples"] == 4
    assert sum(report["augmented_training_zone_counts"]) == 4
    assert report["class_weighting"] == class_weighting
    if class_weighting == "inverse-sqrt":
        expected = inverse_sqrt_class_weights(np.array(report["augmented_training_zone_counts"]))
        np.testing.assert_allclose(report["loss_class_weights"], expected.numpy())
    else:
        assert report["loss_class_weights"] is None
    assert report["best_by_lead_ms"]["100.0"]["examples"] == 2
    assert set(report["train_shot_ids"]).isdisjoint(report["validation_shot_ids"])
    loaded, config, leads = load_model(checkpoint, torch.device("cpu"))
    assert config.freeze_backbone and leads == [100]
    assert loaded(torch.zeros(1, 2, 3, 32, 32)).shape == (1, 6)
