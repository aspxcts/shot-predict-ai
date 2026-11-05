import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from .court import ZONES, zone_for
from .model import ModelConfig
from .players import PlayerCache
from .video import clip_indices, prepare_frame, read_clip


def split_exploratory_videos(project: dict, validation_videos: list[str]):
    shots = project["shots"]
    if len({project["videos"][s["video_id"]]["match_id"] for s in shots}) != 1:
        raise ValueError("Exploratory video holdout requires exactly one labeled match.")
    return split_videos(project, validation_videos)


def split_videos(project: dict, validation_videos: list[str]):
    """Hold out whole clips, from any matches: the shots of one clip never straddle the split.

    Players, courts and broadcasts overlap between the parts, so this is not a new-match evaluation.
    """
    shots = project["shots"]
    held = set(validation_videos)
    labeled = {s["video_id"] for s in shots}
    if not held or held - labeled:
        raise ValueError("Select labeled video IDs for validation.")
    train = [s for s in shots if s["video_id"] not in held]
    validation = [s for s in shots if s["video_id"] in held]
    if not train:
        raise ValueError("Validation must leave at least one training video.")
    train_paths = {Path(project["videos"][s["video_id"]]["path"]).resolve() for s in train}
    val_paths = {Path(project["videos"][s["video_id"]]["path"]).resolve() for s in validation}
    if train_paths & val_paths:
        raise ValueError("Training and validation videos must not reference the same source file.")
    return train, validation, sorted(held)


def augment_clip(clips: torch.Tensor, label: int) -> tuple[torch.Tensor, int]:
    mean = clips.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = clips.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = (clips * std + mean).clamp(0, 1)
    images = TF.adjust_brightness(images, random.uniform(0.85, 1.15))
    images = TF.adjust_contrast(images, random.uniform(0.85, 1.15))
    images = TF.adjust_saturation(images, random.uniform(0.9, 1.1))
    height, width = images.shape[-2:]
    images = TF.affine(
        images, angle=random.uniform(-2, 2),
        translate=[round(random.uniform(-0.025, 0.025) * width),
                   round(random.uniform(-0.025, 0.025) * height)],
        scale=random.uniform(0.95, 1.05), shear=0,
        interpolation=InterpolationMode.BILINEAR, fill=[0.485, 0.456, 0.406],
    )
    if random.random() < 0.5:
        images = images.flip(-1)
        zone = ZONES[label]
        lateral, depth = zone.split("_")
        lateral = {"ad": "deuce", "deuce": "ad", "middle": "middle"}[lateral]
        label = ZONES.index(f"{lateral}_{depth}")
    return ((images - mean) / std).contiguous(), label


class FeatureDataset(Dataset):
    def __init__(self, features: torch.Tensor, labels: torch.Tensor, examples: list[dict]):
        self.features = features
        self.labels = labels
        self.examples = examples

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


def split_matches(project: dict, seed: int, validation_matches: list[str] | None = None):
    shots = project["shots"]
    if not shots:
        raise ValueError("No labels yet. Use the labeling app first.")
    matches = sorted({project["videos"][s["video_id"]]["match_id"] for s in shots})
    if len(matches) < 2:
        raise ValueError("At least TWO labeled matches are needed for a leakage-safe validation split.")
    if validation_matches:
        held_out = set(validation_matches)
        unknown = held_out - set(matches)
        if unknown:
            raise ValueError(f"Unknown or unlabeled validation matches: {sorted(unknown)}")
    else:
        random.Random(seed).shuffle(matches)
        held_out = set(matches[:max(1, math.ceil(len(matches) * 0.2))])
    train = [s for s in shots if project["videos"][s["video_id"]]["match_id"] not in held_out]
    validation = [s for s in shots if project["videos"][s["video_id"]]["match_id"] in held_out]
    if not train or not validation:
        raise ValueError("Both training and validation need labeled matches.")
    return train, validation, sorted(held_out)


class ShotDataset(Dataset):
    def __init__(
        self, project: dict, shots: list[dict], config: ModelConfig, leads: list[float],
        player_cache: Path | None = None,
    ):
        config.validate()
        if not leads or any(lead < 0 for lead in leads):
            raise ValueError("Specify at least one nonnegative prediction lead.")
        self.config = config
        self.project = project
        if config.input_mode == "player_crop" and player_cache is None:
            raise ValueError("This model requires a matching --player-cache.")
        self.player_cache = PlayerCache(player_cache, config) if config.input_mode == "player_crop" else None
        self.examples = []
        for shot in shots:
            video = project["videos"][shot["video_id"]]
            path = Path(video["path"])
            if not path.is_file():
                raise ValueError(f"Missing source video: {path}")
            if shot["hitter"] not in ("near", "far") or shot["zone"] not in ZONES:
                raise ValueError(f"Invalid hitter/zone in shot {shot['id']}")
            if not (0 <= shot["contact_frame"] < shot["bounce_frame"] < video["frames"]):
                raise ValueError(f"Invalid contact/bounce frames in shot {shot['id']}")
            landing_zone = zone_for(*shot["landing"], shot["hitter"])
            if landing_zone != shot["zone"]:
                raise ValueError(f"Inconsistent landing label in shot {shot['id']}")
            for lead in leads:
                try:
                    indices = clip_indices(
                        shot["contact_frame"], video["fps"], config.seconds, config.samples, lead
                    )
                except ValueError as exc:
                    raise ValueError(f"Shot {shot['id']} at {lead} ms: {exc}") from exc
                self.examples.append({
                    "path": path, "indices": indices, "shot": shot, "lead": lead,
                    "label": ZONES.index(shot["zone"]),
                })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]
        if self.player_cache is not None:
            video = self.project["videos"][example["shot"]["video_id"]]
            crops, _ = self.player_cache.load(example, video, self.config)
            # Player crops are upright on both ends; court-oriented full frames retain their old rotation.
            frames = np.stack([prepare_frame(crop, self.config.image_size, "near") for crop in crops])
        else:
            frames = read_clip(
                example["path"], example["indices"], self.config.image_size, example["shot"]["hitter"]
            )
        return torch.from_numpy(frames), example["label"]
