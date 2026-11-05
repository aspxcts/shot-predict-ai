import argparse
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .court import ZONES
from .dataset import FeatureDataset, ShotDataset, augment_clip, split_exploratory_videos, split_matches, split_videos
from .model import ModelConfig, ShotModel
from .storage import SHOT_SOURCES, read_project

LABEL_SOURCES = SHOT_SOURCES


def filter_label_sources(project: dict, sources: list[str]) -> dict:
    """Copy of the project containing only labels from the given sources (labels without a source are manual)."""
    allowed = set(sources)
    shots = [s for s in project["shots"] if s.get("source", "manual") in allowed]
    if not shots:
        raise ValueError(f"No labels come from the selected sources: {sorted(allowed)}.")
    return {**project, "shots": shots}


def choose_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable. Use --device cpu or install a CUDA PyTorch build.")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices are auto, cpu, and cuda.")
    return device


def inverse_sqrt_class_weights(counts: np.ndarray) -> torch.Tensor:
    counts = np.asarray(counts)
    if counts.shape != (len(ZONES),) or not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Class counts must contain six finite, nonnegative values.")
    present = counts > 0
    if not present.any():
        raise ValueError("At least one training class is required.")
    weights = np.zeros(len(ZONES), dtype=np.float32)
    weights[present] = 1 / np.sqrt(counts[present])
    weights[present] /= weights[present].mean()
    return torch.from_numpy(weights)


def cache_features(
    model: ShotModel, dataset: ShotDataset, device: torch.device,
    batch_size: int, augmentation_copies: int = 0,
) -> FeatureDataset:
    model.eval()
    features, targets, examples = [], [], []
    offset = 0
    with torch.no_grad():
        for clips, labels in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            batch_examples = dataset.examples[offset:offset + len(labels)]
            for variant in range(augmentation_copies + 1):
                if variant:
                    augmented = [augment_clip(clip, int(label)) for clip, label in zip(clips, labels)]
                    images = torch.stack([item[0] for item in augmented])
                    variant_labels = torch.tensor([item[1] for item in augmented])
                else:
                    images, variant_labels = clips, labels
                features.append(model.encode(images.to(device)).cpu())
                targets.append(variant_labels)
                examples.extend(
                    {**example, "label": int(label)}
                    for example, label in zip(batch_examples, variant_labels)
                )
            offset += len(labels)
            print(f"Cached {offset}/{len(dataset)} source sequences "
                  f"({augmentation_copies} augmented copies each).", flush=True)
    return FeatureDataset(torch.cat(features), torch.cat(targets), examples)


def evaluate(
    model: ShotModel, loader: DataLoader, device: torch.device, majority: int,
    cached_features: bool = False,
) -> dict:
    model.eval()
    all_probs, all_labels = [], []
    with torch.inference_mode():
        for clips, labels in loader:
            inputs = clips.to(device)
            logits = model.classify(inputs) if cached_features else model(inputs)
            probs = logits.softmax(dim=1).cpu()
            all_probs.append(probs)
            all_labels.append(labels)
    probabilities = torch.cat(all_probs)
    labels = torch.cat(all_labels)
    if not torch.isfinite(probabilities).all():
        raise ValueError("Validation produced non-finite probabilities; training is unstable.")
    predicted = probabilities.argmax(dim=1)
    confusion = torch.zeros((6, 6), dtype=torch.int64)
    for truth, guess in zip(labels.tolist(), predicted.tolist()):
        confusion[truth, guess] += 1
    present = confusion.sum(dim=1) > 0
    recall = confusion.diag()[present] / confusion.sum(dim=1)[present]
    groups = {}
    for field in ("hitter", "shot_type"):
        values = [example["shot"][field] for example in loader.dataset.examples]
        for value in sorted(set(values)):
            mask = torch.tensor([item == value for item in values])
            groups[f"{field}:{value}"] = {
                "examples": int(mask.sum()),
                "top1": float((predicted[mask] == labels[mask]).float().mean()),
            }
    return {
        "examples": len(labels),
        "top1": float((predicted == labels).float().mean()),
        "top2": float((probabilities.topk(2, dim=1).indices == labels[:, None]).any(dim=1).float().mean()),
        "balanced_accuracy_present_classes": float(recall.mean()),
        "training_majority_baseline": float((labels == majority).float().mean()),
        "negative_log_likelihood": float(nn.functional.nll_loss(probabilities.clamp_min(1e-9).log(), labels)),
        "confusion_matrix": confusion.tolist(),
        "by_group": groups,
    }


def players_for(project: dict, shots: list[dict]) -> set[str]:
    players = set()
    for shot in shots:
        video = project["videos"][shot["video_id"]]
        players.update(name for name in (video.get("near_player"), video.get("far_player")) if name)
    return players


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a ResNet18 + GRU pre-contact tennis predictor.")
    p.add_argument("--project", type=Path, default=Path("data") / "project.json")
    p.add_argument("--output", type=Path, default=Path("runs") / "baseline")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--seconds", type=float, default=0.6)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--lead-ms", type=float, nargs="+", default=[100, 200, 300])
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--pretrained", action="store_true", help="Download public ImageNet ResNet18 weights.")
    p.add_argument("--freeze-backbone", action="store_true", help="Requires --pretrained; faster initial training.")
    validation = p.add_mutually_exclusive_group()
    validation.add_argument("--validation-match", action="append", help="Repeat to hold out specific match IDs.")
    validation.add_argument(
        "--exploratory-validation-video", action="append",
        help="Hold out entire video IDs from ONE match. Exploratory only, not match-generalization evidence.",
    )
    validation.add_argument(
        "--validation-video", action="append",
        help="Hold out entire video IDs, from any matches (whole clips, never split). Players, courts and "
             "broadcasts still overlap with training, so this is not a new-match evaluation.",
    )
    p.add_argument("--cache-features", action="store_true", help="Requires a frozen backbone; avoids repeated decoding/encoding.")
    p.add_argument("--augmentation-copies", type=int, default=0,
                   help="Additional augmented training copies per sequence; requires --cache-features.")
    p.add_argument("--class-weighting", choices=("none", "inverse-sqrt"), default="none",
                   help="Optionally give less common training classes more weight in the loss.")
    p.add_argument("--label-sources", nargs="+", choices=LABEL_SOURCES, default=list(LABEL_SOURCES),
                   help="Train and validate only on labels from these sources (default: all). Use 'manual' for "
                        "human-only labels; 'auto_confirmed'/'auto_corrected' come from the review queue.")
    p.add_argument("--player-cache", type=Path, help="Precomputed YOLO hitter cache; trains a player-crop model.")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threads", type=int, default=4)
    return p


def run(args) -> Path:
    if min(args.epochs, args.batch_size, args.threads) < 1 or args.learning_rate <= 0:
        raise ValueError("Epochs, batch size, threads, and learning rate must be positive.")
    if args.freeze_backbone and not args.pretrained:
        raise ValueError("--freeze-backbone requires --pretrained when starting training.")
    if args.cache_features and not args.freeze_backbone:
        raise ValueError("--cache-features requires --freeze-backbone.")
    if args.augmentation_copies < 0 or (args.augmentation_copies and not args.cache_features):
        raise ValueError("Augmentation copies must be nonnegative and require --cache-features.")
    checkpoint_path = args.output / "best.pt"
    if checkpoint_path.exists():
        raise ValueError(f"{checkpoint_path} already exists. Choose another --output to preserve that run.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    project = read_project(args.project)
    project = filter_label_sources(project, args.label_sources)
    validation_videos = []
    scope = "match_holdout"
    if args.exploratory_validation_video:
        train_shots, val_shots, validation_videos = split_exploratory_videos(
            project, args.exploratory_validation_video
        )
        held_out = []
        scope = "exploratory_same_match_video_holdout"
        warnings.warn("EXPLORATORY SAME-MATCH EVALUATION: shared players, court, and broadcast. "
                      "These results do NOT measure generalization to new matches.")
    elif args.validation_video:
        train_shots, val_shots, validation_videos = split_videos(project, args.validation_video)
        held_out = []
        scope = "clip_holdout"
        warnings.warn("CLIP HOLDOUT: validation clips share players, courts and broadcasts with training clips. "
                      "These results do NOT measure generalization to new matches.")
    else:
        train_shots, val_shots, held_out = split_matches(project, args.seed, args.validation_match)
    input_mode, detector_sha256 = "full_frame", ""
    if args.player_cache is not None:
        manifest = json.loads((args.player_cache / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("failures"):
            raise ValueError("Player cache has unresolved detection failures; do not silently omit those shots.")
        input_mode, detector_sha256 = "player_crop", manifest["detector_sha256"]
    config = ModelConfig(
        args.image_size, args.samples, args.seconds, args.hidden, args.freeze_backbone,
        input_mode, detector_sha256,
    )
    train_data = ShotDataset(project, train_shots, config, args.lead_ms, args.player_cache)
    train_counts = np.bincount([e["label"] for e in train_data.examples], minlength=6)
    majority = int(train_counts.argmax())
    if np.any(train_counts == 0):
        warnings.warn(f"Missing training zones: {[ZONES[i] for i in np.where(train_counts == 0)[0]]}")
    overlap = sorted(players_for(project, train_shots) & players_for(project, val_shots))
    if overlap:
        warnings.warn(f"Player overlap across split: {overlap}. This is NOT an unseen-player evaluation.")
    if any(not project["videos"][s["video_id"]].get(side)
           for s in train_shots + val_shots for side in ("near_player", "far_player")):
        warnings.warn("Some player names are missing; unseen-player generalization cannot be verified.")
    model = ShotModel(config, args.pretrained).to(device)
    if args.cache_features:
        train_data = cache_features(
            model, train_data, device, args.batch_size, args.augmentation_copies
        )
    val_loaders = {}
    for lead in sorted(set(args.lead_ms)):
        dataset = ShotDataset(project, val_shots, config, [lead], args.player_cache)
        if args.cache_features:
            dataset = cache_features(model, dataset, device, args.batch_size)
        val_loaders[str(lead)] = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.learning_rate
    )
    optimized_counts = np.bincount([e["label"] for e in train_data.examples], minlength=6)
    loss_weights = (
        inverse_sqrt_class_weights(optimized_counts).to(device)
        if args.class_weighting == "inverse-sqrt" else None
    )
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    args.output.mkdir(parents=True, exist_ok=True)
    history = []
    best_loss = float("inf")
    best_epoch = None
    best_metrics = None
    print(f"Device: {device}; train shots: {len(train_shots)}; validation shots: {len(val_shots)}; "
          f"evaluation: {scope}; held-out matches: {held_out}; "
          f"held-out videos: {validation_videos}", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for clips, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            inputs = clips.to(device)
            logits = model.classify(inputs) if args.cache_features else model(inputs)
            loss = criterion(logits, labels.to(device))
            if not torch.isfinite(loss):
                raise ValueError("Training loss is non-finite. Check inputs and learning rate.")
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
        metrics = {
            lead: evaluate(model, loader, device, majority, args.cache_features)
            for lead, loader in val_loaders.items()
        }
        average_loss = sum(m["negative_log_likelihood"] for m in metrics.values()) / len(metrics)
        entry = {"epoch": epoch, "train_loss": total_loss / len(train_data), "by_lead_ms": metrics}
        history.append(entry)
        print(json.dumps(entry), flush=True)
        if average_loss < best_loss:
            best_loss = average_loss
            best_epoch = epoch
            best_metrics = metrics
            torch.save({
                "version": 1, "config": config.to_dict(), "zones": list(ZONES),
                "state_dict": model.state_dict(), "lead_ms": args.lead_ms,
                "validation_matches": held_out, "epoch": epoch,
                "evaluation_scope": scope, "validation_video_ids": validation_videos,
            }, checkpoint_path)
        report = {
            "zones": list(ZONES), "config": config.to_dict(),
            "validation_matches": held_out, "player_overlap": overlap,
            "train_shots": len(train_shots), "validation_shots": len(val_shots),
            "training_zone_counts": train_counts.tolist(), "seed": args.seed,
            "lead_ms": args.lead_ms,
            "train_shot_ids": [s["id"] for s in train_shots],
            "validation_shot_ids": [s["id"] for s in val_shots],
            "best_validation_nll": best_loss, "history": history,
            "best_epoch": best_epoch, "best_by_lead_ms": best_metrics,
            "evaluation_scope": scope, "validation_video_ids": validation_videos,
            "augmentation_copies": args.augmentation_copies,
            "cached_features": args.cache_features,
            "pretrained_backbone": args.pretrained,
            "learning_rate": args.learning_rate, "batch_size": args.batch_size,
            "class_weighting": args.class_weighting, "label_sources": sorted(args.label_sources),
            "loss_class_weights": loss_weights.cpu().tolist() if loss_weights is not None else None,
            "training_project": str(args.project.resolve()),
            "player_cache": str(args.player_cache.resolve()) if args.player_cache is not None else None,
            "training_examples": len(train_data),
            "augmented_training_zone_counts": optimized_counts.tolist(),
        }
        (args.output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved checkpoint: {checkpoint_path}", flush=True)
    return checkpoint_path


def main():
    p = parser()
    args = p.parse_args()
    try:
        run(args)
    except (ValueError, OSError) as exc:
        p.exit(1, f"Training failed: {exc}\n")


if __name__ == "__main__":
    main()
