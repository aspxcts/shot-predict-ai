import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from .court import ZONES
from .dataset import ShotDataset, augment_clip
from .model import ModelConfig, ShotModel
from .predict import load_model
from .storage import read_project


@dataclass
class GradCAM:
    probabilities: torch.Tensor
    target: int
    native: torch.Tensor
    maps: torch.Tensor


def normalize_maps(positive: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if not torch.isfinite(positive).all() or (positive < 0).any():
        raise ValueError("Grad-CAM maps must be finite and nonnegative.")
    peak = positive.max()
    normalized = positive / peak if peak > 0 else torch.zeros_like(positive)
    return F.interpolate(
        normalized[:, None], size=size, mode="bilinear", align_corners=False
    )[:, 0]


def gradcam(model: ShotModel, clip: torch.Tensor, target: int | None = None) -> GradCAM:
    if clip.ndim != 4 or clip.shape[1] != 3:
        raise ValueError("Expected one sequence with shape [frames, 3, height, width].")
    if model.training:
        raise ValueError("Put the model in evaluation mode before explaining predictions.")
    if torch.is_inference_mode_enabled():
        raise ValueError("Grad-CAM needs autograd; call outside torch.inference_mode().")
    if target is not None and not 0 <= target < len(ZONES):
        raise ValueError("Invalid Grad-CAM target class.")
    activations = []
    hook = model.backbone.layer4.register_forward_hook(
        lambda _module, _inputs, output: activations.append(output)
    )
    try:
        with torch.enable_grad():
            # Input gradients keep the frozen CNN in the autograd graph.
            inputs = clip.detach().clone().unsqueeze(0).requires_grad_(True)
            logits = model(inputs)
            if not torch.isfinite(logits).all():
                raise ValueError("Model produced non-finite logits.")
            chosen = int(logits[0].argmax()) if target is None else target
            features, = activations
            gradients, = torch.autograd.grad(logits[0, chosen], features)
            weights = gradients.mean(dim=(-2, -1), keepdim=True)
            positive = (weights * features).sum(dim=1).relu().detach()
            maps = normalize_maps(positive, tuple(clip.shape[-2:]))
            return GradCAM(
                logits[0].softmax(0).detach().cpu(), chosen,
                positive.cpu(), maps.cpu(),
            )
    finally:
        hook.remove()


def input_images(clip: torch.Tensor) -> list[np.ndarray]:
    mean = clip.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = clip.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    rgb = ((clip.detach() * std + mean).clamp(0, 1) * 255).round()
    rgb = rgb.to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
    return [cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in rgb]


def overlay(image: np.ndarray, heat: np.ndarray) -> np.ndarray:
    colors = cv2.applyColorMap(np.round(heat * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    alpha = (heat * 0.65)[..., None]
    return np.round(image * (1 - alpha) + colors * alpha).astype(np.uint8)


def save_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write image: {path}")


def sequence_sheet(
    rows: list[tuple[str, list[np.ndarray]]], titles: list[str], headings: list[str],
    footer: str = "", cell: int = 224,
) -> np.ndarray:
    row_height = cell + 30
    header = 28 * (len(titles) + 1)
    sheet = np.full((header + len(rows) * row_height + 35, cell * len(headings), 3), 24, np.uint8)
    for i, title in enumerate(titles):
        cv2.putText(sheet, title, (8, 24 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, .5, (235, 235, 235), 1)
    for i, heading in enumerate(headings):
        cv2.putText(sheet, heading, (i * cell + 8, header - 8), cv2.FONT_HERSHEY_SIMPLEX, .48, (200, 200, 200), 1)
    for row_index, (name, images) in enumerate(rows):
        y = header + row_index * row_height
        cv2.putText(sheet, name, (8, y + 20), cv2.FONT_HERSHEY_SIMPLEX, .46, (220, 220, 220), 1)
        for column, image in enumerate(images):
            x = column * cell
            sheet[y + 30:y + 30 + cell, x:x + cell] = cv2.resize(
                image, (cell, cell), interpolation=cv2.INTER_NEAREST
            )
    cv2.putText(sheet, footer, (8, sheet.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, .43, (210, 210, 210), 1)
    return sheet


def select_shots(project: dict, shot_ids: list[str]) -> list[dict]:
    by_id = {shot["id"]: shot for shot in project["shots"]}
    missing = set(shot_ids) - by_id.keys()
    if missing:
        raise ValueError(f"Run snapshot is missing shot IDs: {sorted(missing)}")
    return sorted(
        [by_id[key] for key in shot_ids],
        key=lambda shot: (project["videos"][shot["video_id"]]["path"], shot["contact_frame"]),
    )


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Explain a saved run using held-out, sequence-level Grad-CAM.")
    p.add_argument("--run", type=Path, default=Path("runs") / "exploratory-121")
    p.add_argument("--output", type=Path, help="New folder; defaults to RUN/explanations.")
    p.add_argument("--limit", type=int, default=0, help="Explain the first N held-out shots; 0 means all.")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--player-cache", type=Path, help="Override the saved player-cache location for a crop model.")
    return p


def run(args) -> Path:
    if args.limit < 0 or args.threads < 1:
        raise ValueError("Limit must be nonnegative and threads must be positive.")
    output = args.output if args.output is not None else args.run / "explanations"
    if output.exists():
        raise ValueError(f"{output} already exists. Choose a new --output.")
    report = json.loads((args.run / "metrics.json").read_text(encoding="utf-8"))
    project = read_project(args.run / "project_snapshot.json")
    if set(report["train_shot_ids"]) & set(report["validation_shot_ids"]):
        raise ValueError("Training and validation shot IDs overlap.")
    torch.set_num_threads(args.threads)
    model, config, leads = load_model(args.run / "best.pt", torch.device("cpu"))
    if config.to_dict() != ModelConfig(**report["config"]).to_dict() or leads != report["lead_ms"]:
        raise ValueError("The checkpoint and run report do not describe the same inputs.")
    shots = select_shots(project, report["validation_shot_ids"])
    if args.limit:
        shots = shots[:args.limit]
    if not shots:
        raise ValueError("This run has no held-out shots to explain.")
    cache = args.player_cache
    if cache is None and report.get("player_cache"):
        cache = Path(report["player_cache"])
    data = ShotDataset(project, shots, config, leads, cache)
    training = select_shots(project, report["train_shot_ids"])
    if not training:
        raise ValueError("This run has no training shots to show.")
    training_data = ShotDataset(project, training[:4], config, [leads[0]], cache)
    output.mkdir(parents=True)
    records, native_maps, overview = [], [], []
    for i, example in enumerate(data.examples):
        clip, truth = data[i]
        result = gradcam(model, clip)
        images = input_images(clip)
        overlays = [overlay(image, heat.numpy()) for image, heat in zip(images, result.maps)]
        shot = example["shot"]
        fps = project["videos"][shot["video_id"]]["fps"]
        headings = [
            f"Frame {frame} | {(frame - shot['contact_frame']) / fps * 1000:.0f} ms"
            for frame in example["indices"]
        ]
        filename = f"shot_{i + 1:03d}_{shot['id']}.jpg"
        predicted = ZONES[result.target]
        peak = result.native.flatten(1).max(dim=1).values
        title = [
            f"Held-out #{i + 1} | {shot['hitter']} | actual: {ZONES[truth]} | predicted: {predicted}",
            "Predicted-class Grad-CAM through the entire CNN + GRU; positive evidence only.",
        ]
        if not bool((peak > 0).any()):
            title.append("No positive Grad-CAM signal for this sequence.")
        sheet = sequence_sheet(
            [("Actual model inputs (no augmentation)", images), ("Grad-CAM overlay", overlays)],
            title, headings,
            f"Native grid: {result.native.shape[-2]}x{result.native.shape[-1]}. Shared scale across frames; not calibrated attention.",
        )
        save_image(output / filename, sheet)
        if i < 8:
            overview.extend([
                (f"#{i + 1} {shot['hitter']} | actual {ZONES[truth]} | predicted {predicted}",
                 [images[-1], overlays[-1]]),
            ])
        records.append({
            "shot_id": shot["id"], "video_id": shot["video_id"], "hitter": shot["hitter"],
            "contact_frame": shot["contact_frame"], "input_frames": example["indices"],
            "lead_ms": example["lead"], "actual_zone": ZONES[truth], "target_zone": predicted,
            "probabilities": result.probabilities.tolist(), "image": filename,
            "positive_peak_by_frame": peak.tolist(), "no_positive_signal": not bool((peak > 0).any()),
        })
        native_maps.append(result.native.numpy())
        print(f"Explained {i + 1}/{len(data)} held-out sequences.", flush=True)
    save_image(output / "overview.jpg", sequence_sheet(
        overview, ["First held-out examples in time order; final input frame only.",
                   "Warm colors: stronger positive Grad-CAM for predicted class."],
        ["Model input", "Heatmap overlay"],
        "Per-sequence scale; validation was used for checkpoint selection.",
        cell=320,
    ))
    training_rows = []
    for i, example in enumerate(training_data.examples):
        clip, label = training_data[i]
        training_rows.append((
            f"Train #{i + 1} | {example['shot']['hitter']} | {ZONES[label]}",
            input_images(clip),
        ))
    input_description = (
        ["Original training inputs: YOLO hitter crops with racket/body context.",
         f"Resized to {config.image_size}x{config.image_size}; both hitters remain upright."]
        if config.input_mode == "player_crop" else
        ["Original training inputs: full broadcast frames, not player/ball crops.",
         f"Resized to {config.image_size}x{config.image_size}; far-hitter inputs rotated 180 degrees."]
    )
    save_image(output / "training_inputs.jpg", sequence_sheet(
        training_rows, input_description,
        [f"Input {i + 1}" for i in range(config.samples)],
        "Display is enlarged. RGB ImageNet normalization is reversed here for viewing.",
    ))
    clip, label = training_data[0]
    augmentation_rows = [("Original", input_images(clip))]
    random_state = random.getstate()
    try:
        random.seed(report["seed"])
        for i in range(min(report.get("augmentation_copies", 0), 3)):
            augmented, augmented_label = augment_clip(clip, label)
            augmentation_rows.append((
                f"Illustrative augmentation #{i + 1}: {ZONES[augmented_label]}", input_images(augmented)
            ))
    finally:
        random.setstate(random_state)
    save_image(output / "augmentation_examples.jpg", sequence_sheet(
        augmentation_rows,
        ["Augmentation policy examples; NOT the exact cached copies from training.",
         "Same transform across frames. Mirroring swaps ad/deuce labels."],
        [f"Input {i + 1}" for i in range(config.samples)],
    ))
    np.savez_compressed(output / "native_gradcam.npz", maps=np.stack(native_maps))
    manifest = {
        "checkpoint": str((args.run / "best.pt").resolve()), "config": config.to_dict(),
        "evaluation_scope": report.get("evaluation_scope", "match_holdout"),
        "method": "Grad-CAM of the predicted logit at ResNet layer4, through the temporal GRU.",
        "normalization": "One shared maximum over all frames in each sequence; positive contributions only.",
        "limitations": [
            "Coarse spatial evidence, not ball tracking, calibrated attention, or a causal explanation.",
            "Heatmap scales cannot be compared between different sequences.",
            "Held-out validation was used for checkpoint selection; this is not an untouched test set.",
            "Augmentation images illustrate the policy, not the exact cached training copies.",
        ],
        "examples": records,
        "training_examples": [
            {"shot_id": e["shot"]["id"], "input_frames": e["indices"], "lead_ms": e["lead"]}
            for e in training_data.examples
        ],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved explanations to {output}", flush=True)
    return output


def main():
    p = parser()
    args = p.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as exc:
        p.exit(1, f"Explanation failed: {exc}\n")


if __name__ == "__main__":
    main()
