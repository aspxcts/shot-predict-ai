import csv
import hashlib
import json
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from shot_predict.court import Court
from shot_predict.dataset import ShotDataset
from shot_predict.explain import parser as explain_parser, run as explain
from shot_predict.model import ModelConfig, ShotModel
from shot_predict.players import (
    CROP_VERSION, PlayerCache, PlayerCropper, PlayerNotFoundError,
    crop_sequence, input_spec, select_player, spec_key,
)
from shot_predict.predict import load_model, parser as export_parser, run as export
from shot_predict.storage import add_shot, write_project
from shot_predict.train import parser as train_parser, run as train
from shot_predict.video import metadata, prepare_frame, read_frames


def test_selects_annotated_end_not_the_largest_person():
    court = Court([[100, 900], [900, 900], [900, 100], [100, 100]])
    near = [400, 680, 480, 980, .9]
    far = [500, 40, 550, 180, .9]
    official = [1100, 100, 1500, 900, .99]
    boxes = np.array([official, far, near])
    assert select_player(boxes, "near", court) == near
    assert select_player(boxes, "far", court) == far
    with pytest.raises(PlayerNotFoundError, match="far-side"):
        select_player(np.array([near]), "far", court)
    with pytest.raises(PlayerNotFoundError):
        select_player(np.empty((0, 5)), "near", court)


def test_square_crops_follow_player_with_stable_scale_and_edge_padding():
    court = Court([[20, 110], [140, 110], [140, 10], [20, 10]])
    frames = [np.full((120, 160, 3), 200, np.uint8) for _ in range(2)]
    boxes = [np.array([[40, 40, 70, 100, .9]]), np.array([[60, 30, 95, 105, .9]])]
    crops, info = crop_sequence(frames, boxes, "near", court, 64)
    for crop_box, detections in zip(info["crop_boxes"], boxes):
        x1, y1, x2, y2 = crop_box
        left, top, right, bottom, _ = detections[0]
        assert x2 - x1 == y2 - y1 == 120
        assert x1 < left and y1 < top and x2 > right and y2 > bottom
        assert abs((x1 + x2 - left - right) / 2) <= 1
        assert abs((y1 + y2 - top - bottom) / 2) <= 1
    assert crops.shape == (2, 64, 64, 3)
    assert crops.dtype == np.uint8
    assert info["crop_boxes"][0] != info["crop_boxes"][1]
    assert (crops == [104, 116, 124]).all(axis=-1).any()


def test_yolo_is_local_person_only_and_checks_detector_identity(tmp_path, monkeypatch):
    import shot_predict.players as players

    monkeypatch.setattr(players, "ROOT", tmp_path)
    weights = tmp_path / "data" / "models" / "yolo11n.pt"
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"synthetic-detector")
    settings = {}

    class Detector:
        def __init__(self, path):
            assert path == str(weights)
            assert players.os.environ["YOLO_OFFLINE"] == "true"
            assert players.os.environ["YOLO_AUTOINSTALL"] == "false"

        def predict(self, frames, **kwargs):
            assert kwargs["classes"] == [0] and kwargs["save"] is False
            assert kwargs["device"] == "cpu"
            return [SimpleNamespace(boxes=SimpleNamespace(
                xyxy=torch.tensor([[1., 2., 3., 4.]]), conf=torch.tensor([.9])
            )) for _ in frames]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=Detector, settings=settings))
    detector = PlayerCropper()
    assert settings["sync"] is False and settings["wandb"] is False
    assert detector.detect([np.zeros((10, 10, 3), np.uint8)])[0].shape == (1, 5)
    with pytest.raises(ValueError, match="do not match"):
        PlayerCropper("0" * 64)


def test_a_frame_the_batch_dropped_is_retried_alone_and_truly_empty_frames_stay_empty():
    calls = []

    def result(boxes):
        return SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.tensor(boxes).reshape(-1, 4), conf=torch.tensor([.9] * len(boxes))))

    class Model:
        def predict(self, frames, **kwargs):
            calls.append(len(frames))
            if len(frames) > 1:  # the post-processing time limit tripped: only the first frame got its detections
                return [result([[1., 2., 3., 4.]])] + [result([]) for _ in frames[1:]]
            return [result([] if float(frames[0][0, 0, 0]) == 255 else [[5., 6., 7., 8.]])]

    detector = PlayerCropper.__new__(PlayerCropper)
    detector.model = Model()
    ordinary = np.zeros((10, 10, 3), np.uint8)
    nobody = np.full((10, 10, 3), 255, np.uint8)
    found = detector.detect([ordinary, ordinary, nobody])
    assert [len(f) for f in found] == [1, 1, 0]
    assert found[1][0, :4].tolist() == [5., 6., 7., 8.]  # recovered by the retry, not by the batch
    assert calls == [3, 1, 1]  # one batch, then each empty frame once; a single frame is never retried
    calls.clear()
    assert [len(f) for f in detector.detect([nobody])] == [0] and calls == [1]


@pytest.fixture
def cropped_project(tmp_path):
    court = Court([[20, 110], [140, 110], [140, 10], [20, 10]])
    config = ModelConfig(64, 2, .1, 8, True, "player_crop", "0" * 64)
    project = {"version": 1, "videos": {}, "shots": []}
    for key in ("first", "last"):
        source = tmp_path / f"{key}.avi"
        writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 20, (160, 120))
        assert writer.isOpened()
        for i in range(80):
            image = np.full((120, 160, 3), i * 3, np.uint8)
            cv2.rectangle(image, (40, 40), (70, 100), (20, 200, 30), -1)
            cv2.rectangle(image, (75, 0), (100, 8), (200, 20, 20), -1)
            writer.write(image)
        writer.release()
        project["videos"][key] = {
            "path": str(source), "match_id": "one-match", "corners": court.corners,
            "near_player": "near", "far_player": "far", **metadata(source),
        }
        add_shot(project, key, 30, 42, "near", "groundstroke", court.target("ad_deep", "near"))
        add_shot(project, key, 50, 65, "far", "groundstroke", court.target("deuce_short", "far"))
    indexing = ModelConfig(64, 2, .1, 8, True)
    data = ShotDataset(project, project["shots"], indexing, [100])
    folder = tmp_path / "cache"
    folder.mkdir()
    manifest = {"version": CROP_VERSION, "detector_sha256": config.detector_sha256, "entries": {}, "failures": []}
    for example in data.examples:
        video = project["videos"][example["shot"]["video_id"]]
        frames = read_frames(example["path"], example["indices"])
        detections = [np.array([[40, 40, 70, 100, .9], [80, 5, 95, 20, .9]]) for _ in frames]
        crops, info = crop_sequence(frames, detections, example["shot"]["hitter"], court, 64)
        spec = input_spec(example, video, config)
        key = spec_key(spec)
        np.savez_compressed(folder / f"{key}.npz", frames=crops)
        manifest["entries"][key] = {"spec": spec, **info}
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return project, config, folder


def test_player_cache_uses_no_landing_labels_and_keeps_far_players_upright(cropped_project):
    project, config, folder = cropped_project
    data = ShotDataset(project, project["shots"], config, [100], folder)
    example = data.examples[1]
    video = project["videos"][example["shot"]["video_id"]]
    crops, _ = data.player_cache.load(example, video, config)
    tensor, _ = data[1]
    np.testing.assert_allclose(tensor[0].numpy(), prepare_frame(crops[0], 64, "near"))
    assert not np.allclose(tensor[0].numpy(), prepare_frame(crops[0], 64, "far"))
    changed = {**example, "shot": {**example["shot"], "zone": "middle_deep", "landing": [0.2, 0.8]}}
    assert spec_key(input_spec(example, video, config)) == spec_key(input_spec(changed, video, config))
    with pytest.raises(ValueError, match="matching --player-cache"):
        ShotDataset(project, project["shots"], config, [100])
    with pytest.raises(ValueError, match="rebuild"):
        data.player_cache.load({**example, "indices": [40, 41]}, video, config)
    wrong = ModelConfig(64, 2, .1, 8, True, "player_crop", "1" * 64)
    with pytest.raises(ValueError, match="do not match"):
        PlayerCache(folder, wrong)


def test_cropped_training_explanation_and_missing_player_export(cropped_project, tmp_path, monkeypatch):
    import shot_predict.train as training
    import shot_predict.predict as prediction

    torch.set_num_threads(1)
    project, config, folder = cropped_project
    project_path = tmp_path / "project.json"
    write_project(project_path, project)
    monkeypatch.setattr(training, "ShotModel", lambda config, pretrained: ShotModel(config))
    output = tmp_path / "run"
    args = train_parser().parse_args([
        "--project", str(project_path), "--output", str(output),
        "--image-size", "64", "--samples", "2", "--seconds", ".1", "--hidden", "8",
        "--epochs", "1", "--threads", "1", "--batch-size", "2", "--lead-ms", "100",
        "--pretrained", "--freeze-backbone", "--cache-features", "--augmentation-copies", "1",
        "--exploratory-validation-video", "last", "--player-cache", str(folder),
    ])
    with pytest.warns(UserWarning):
        checkpoint = train(args)
    report = json.loads((output / "metrics.json").read_text())
    assert report["config"]["input_mode"] == "player_crop"
    assert report["training_examples"] == 4
    _, saved_config, _ = load_model(checkpoint, torch.device("cpu"))
    assert saved_config == config
    write_project(output / "project_snapshot.json", project)
    explanation = explain(explain_parser().parse_args(["--run", str(output), "--limit", "1", "--threads", "1"]))
    assert (explanation / "training_inputs.jpg").is_file()
    assert json.loads((explanation / "manifest.json").read_text())["config"]["input_mode"] == "player_crop"

    class FakeDetector:
        def __init__(self, expected):
            assert expected == config.detector_sha256
            self.calls = 0

        def detect(self, frames):
            self.calls += 1
            boxes = (np.array([[40, 40, 70, 100, .9], [80, 5, 95, 20, .9]])
                     if self.calls < 3 else np.empty((0, 5)))
            return [boxes for _ in frames]

    monkeypatch.setattr(prediction, "PlayerCropper", FakeDetector)
    video, csv_path = export(export_parser().parse_args([
        "--project", str(project_path), "--video-id", "last", "--checkpoint", str(checkpoint),
        "--output", str(tmp_path / "cropped.mp4"), "--stride", "10", "--threads", "1",
    ]))
    assert video.is_file()
    with csv_path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    assert any(row["status"] == "ok" for row in rows)
    missing = [row for row in rows if row["status"] != "ok"]
    assert missing and all(row["ad_deep"] == "" for row in missing)


def test_old_checkpoint_config_keeps_full_frame_defaults():
    legacy = {"image_size": 160, "samples": 4, "seconds": .6, "hidden": 64, "freeze_backbone": True}
    config = ModelConfig(**legacy)
    assert config.input_mode == "full_frame" and config.detector_sha256 == ""
