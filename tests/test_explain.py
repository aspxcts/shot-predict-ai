import hashlib
import json

import cv2
import numpy as np
import pytest
import torch

from shot_predict.court import Court, ZONES
from shot_predict.explain import gradcam, input_images, normalize_maps, parser, run
from shot_predict.model import ModelConfig, ShotModel
from shot_predict.storage import add_shot, write_project
from shot_predict.video import metadata


def test_gradcam_traces_frozen_cnn_and_temporal_head_without_changing_model():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    model = ShotModel(ModelConfig(image_size=64, samples=4, hidden=8, freeze_backbone=True))
    model.eval()
    clip = torch.randn(4, 3, 64, 64)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    hook_count = len(model.backbone.layer4._forward_hooks)
    with torch.no_grad():
        expected = model(clip[None])[0].softmax(0)
    result = gradcam(model, clip)
    torch.testing.assert_close(result.probabilities, expected)
    assert result.target == int(expected.argmax())
    assert result.native.shape == (4, 2, 2)
    assert result.maps.shape == (4, 64, 64)
    assert torch.isfinite(result.maps).all()
    assert (result.maps >= 0).all() and (result.maps <= 1).all()
    assert len(model.backbone.layer4._forward_hooks) == hook_count
    assert all(parameter.grad is None for parameter in model.parameters())
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    with torch.no_grad():
        model.head.weight[1].copy_(-model.head.weight[0])
        model.head.bias[:2].zero_()
    first = gradcam(model, clip, target=0)
    second = gradcam(model, clip, target=1)
    assert not torch.equal(first.native, second.native)


def test_gradcam_normalization_shares_one_scale_across_time():
    maps = torch.tensor([[[1., 0.], [0., 0.]], [[2., 0.], [0., 0.]]])
    normalized = normalize_maps(maps, (2, 2))
    assert normalized[0].max() == 0.5
    assert normalized[1].max() == 1
    assert torch.equal(normalize_maps(torch.zeros_like(maps), (2, 2)), torch.zeros_like(maps))
    with pytest.raises(ValueError, match="finite and nonnegative"):
        normalize_maps(torch.full_like(maps, float("nan")), (2, 2))


def test_input_display_reverses_normalization_without_flipping_orientation():
    raw = torch.zeros(2, 3, 32, 32)
    raw[:, 0, :, :16] = 1
    mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1)
    images = input_images((raw - mean) / std)
    np.testing.assert_array_equal(images[0][0, 0], [0, 0, 255])
    np.testing.assert_array_equal(images[0][0, -1], [0, 0, 0])


def test_explanation_export_uses_saved_validation_ids_and_exact_inputs(tmp_path):
    torch.set_num_threads(1)
    source = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"MJPG"), 20, (160, 120))
    assert writer.isOpened()
    for i in range(80):
        image = np.full((120, 160, 3), i * 3, np.uint8)
        cv2.rectangle(image, (10, 10), (40, 100), (0, 0, 200), -1)
        writer.write(image)
    writer.release()
    court = Court([[20, 110], [140, 110], [140, 10], [20, 10]])
    project = {
        "version": 1,
        "videos": {"v": {"path": str(source), "match_id": "match", "corners": court.corners,
                         **metadata(source)}},
        "shots": [],
    }
    training = add_shot(project, "v", 30, 42, "near", "groundstroke", court.target("ad_deep", "near"))
    validation = add_shot(project, "v", 50, 65, "far", "groundstroke", court.target("deuce_short", "far"))
    config = ModelConfig(image_size=64, samples=4, seconds=.1, hidden=8, freeze_backbone=True)
    model = ShotModel(config)
    saved_run = tmp_path / "run"
    saved_run.mkdir()
    checkpoint = saved_run / "best.pt"
    torch.save({"version": 1, "config": config.to_dict(), "zones": list(ZONES),
                "state_dict": model.state_dict(), "lead_ms": [100.]}, checkpoint)
    write_project(saved_run / "project_snapshot.json", project)
    report = {
        "config": config.to_dict(), "lead_ms": [100.], "seed": 42, "augmentation_copies": 1,
        "train_shot_ids": [training["id"]], "validation_shot_ids": [validation["id"]],
        "evaluation_scope": "synthetic_test",
    }
    (saved_run / "metrics.json").write_text(json.dumps(report), encoding="utf-8")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    args = parser().parse_args(["--run", str(saved_run), "--threads", "1"])
    output = run(args)
    manifest = json.loads((output / "manifest.json").read_text())
    example, = manifest["examples"]
    assert example["shot_id"] == validation["id"]
    assert example["input_frames"] == [46, 47, 47, 48]
    assert max(example["input_frames"]) < validation["contact_frame"]
    assert manifest["training_examples"][0]["shot_id"] == training["id"]
    assert np.load(output / "native_gradcam.npz")["maps"].shape == (1, 4, 2, 2)
    for name in ("overview.jpg", "training_inputs.jpg", "augmentation_examples.jpg", example["image"]):
        assert cv2.imread(str(output / name)) is not None
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == digest
    with pytest.raises(ValueError, match="already exists"):
        run(args)
