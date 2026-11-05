import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import torch

from .court import Court
from .model import ModelConfig
from .storage import input_corners, read_project
from .video import prepare_frame, read_frames

ROOT = Path(__file__).resolve().parents[1]
CROP_VERSION = 2
DETECTOR_NAME = "yolo11n.pt"
DETECTOR_SIZE = 960
CROP_SCALE = 1.6


class PlayerNotFoundError(ValueError):
    pass


def select_player(boxes: np.ndarray, hitter: str, court: Court, previous=None) -> list[float]:
    if hitter not in ("near", "far"):
        raise ValueError("Hitter must be near or far.")
    if boxes.ndim != 2 or boxes.shape[1] != 5 or not np.isfinite(boxes).all():
        raise ValueError("Expected finite person detections [x1, y1, x2, y2, confidence].")
    candidates = []
    for box in boxes:
        x1, y1, x2, y2, confidence = map(float, box)
        if x2 <= x1 or y2 <= y1 or not 0 <= confidence <= 1:
            raise ValueError("Detector returned an invalid person box.")
        x, y = court.normalized((x1 + x2) / 2, y2)
        on_side = 0.5 < y <= 1.8 if hitter == "near" else -0.8 <= y < 0.5
        if not on_side or not -0.45 <= x <= 1.45:
            continue
        lateral = max(0, -x, x - 1)
        baseline = abs(y - (1 if hitter == "near" else 0))
        score = confidence * math.sqrt((x2 - x1) * (y2 - y1)) / (1 + 4 * lateral + baseline)
        if previous is not None:
            px1, py1, px2, py2, _ = previous
            distance = math.hypot((x1 + x2 - px1 - px2) / 2, y2 - py2)
            score /= 1 + distance / max(1, 2 * (py2 - py1))
        candidates.append((score, list(map(float, box))))
    if not candidates:
        raise PlayerNotFoundError(f"No plausible {hitter}-side player; check visibility and court calibration.")
    return max(candidates, key=lambda item: item[0])[1]


def crop_sequence(
    frames: list[np.ndarray], detections: list[np.ndarray], hitter: str, court: Court, size: int,
) -> tuple[np.ndarray, dict]:
    if not frames or len(frames) != len(detections):
        raise ValueError("Each sequence frame needs person detections.")
    selected = []
    for boxes in detections:
        selected.append(select_player(boxes, hitter, court, selected[-1] if selected else None))
    boxes = np.asarray(selected)
    # Keep scale stable without making running players tiny in a sequence-wide union.
    side = math.ceil(np.max(boxes[:, 2:4] - boxes[:, :2]) * CROP_SCALE)
    crops, crop_boxes = [], []
    for frame, box in zip(frames, boxes):
        x = math.floor((box[0] + box[2] - side) / 2)
        y = math.floor((box[1] + box[3] - side) / 2)
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = max(0, x), max(0, y), min(width, x + side), min(height, y + side)
        if x1 >= x2 or y1 >= y2:
            raise PlayerNotFoundError("Selected player crop does not intersect the image.")
        square = np.empty((side, side, 3), dtype=np.uint8)
        square[:] = (104, 116, 124)
        square[y1 - y:y2 - y, x1 - x:x2 - x] = frame[y1:y2, x1:x2]
        crops.append(cv2.resize(square, (size, size)))
        crop_boxes.append([x, y, x + side, y + side])
    return np.stack(crops), {"person_boxes": selected, "crop_boxes": crop_boxes}


class PlayerCropper:
    def __init__(self, expected_sha256: str = ""):
        weights = ROOT / "data" / "models" / DETECTOR_NAME
        if not weights.exists():
            weights.parent.mkdir(parents=True, exist_ok=True)
            temporary = weights.with_suffix(".download")
            try:
                torch.hub.download_url_to_file(
                    f"https://github.com/ultralytics/assets/releases/download/v8.3.0/{DETECTOR_NAME}",
                    str(temporary), progress=False,
                )
                temporary.replace(weights)
            finally:
                temporary.unlink(missing_ok=True)
        self.sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
        if expected_sha256 and self.sha256 != expected_sha256:
            raise ValueError("YOLO detector weights do not match this classifier's preprocessing.")
        config_dir = ROOT / "data" / ".ultralytics"
        config_dir.mkdir(parents=True, exist_ok=True)
        os.environ["YOLO_CONFIG_DIR"] = str(config_dir)
        os.environ["YOLO_OFFLINE"] = "true"
        os.environ["YOLO_AUTOINSTALL"] = "false"
        try:
            from ultralytics import YOLO, settings
        except ImportError as exc:
            raise ValueError('Install local YOLO support: python -m pip install -e ".[players]"') from exc
        settings.update({key: False for key in (
            "sync", "clearml", "comet", "dvc", "mlflow", "neptune",
            "raytune", "tensorboard", "wandb",
        )})
        self.model = YOLO(str(weights))

    def _predict(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        results = self.model.predict(
            frames, classes=[0], conf=0.2, imgsz=DETECTOR_SIZE, max_det=40,
            device="cpu", verbose=False, save=False,
        )
        return [
            np.column_stack((result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()))
            for result in results
        ]

    def detect(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        boxes = self._predict(frames)
        # On a CPU shared with other jobs the detector's post-processing time limit can trip and silently leave the
        # rest of a batch without detections, so an empty frame is retried on its own before it counts as empty.
        if len(frames) > 1:
            for index, found in enumerate(boxes):
                if len(found) == 0:
                    boxes[index] = self._predict([frames[index]])[0]
        return boxes


def input_spec(example: dict, video: dict, config: ModelConfig) -> dict:
    source = Path(example["path"]).resolve()
    stat = source.stat()
    return {
        "version": CROP_VERSION, "source": str(source), "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns, "indices": example["indices"],
        "hitter": example["shot"]["hitter"], "corners": input_corners(example["shot"], video),
        "image_size": config.image_size, "detector_sha256": config.detector_sha256,
        "detector_size": DETECTOR_SIZE, "crop_scale": CROP_SCALE,
    }


def spec_key(spec: dict) -> str:
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()


class PlayerCache:
    def __init__(self, folder: Path, config: ModelConfig):
        self.folder = folder
        self.manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
        if (self.manifest.get("version") != CROP_VERSION
                or self.manifest.get("detector_sha256") != config.detector_sha256):
            raise ValueError("Player cache and model preprocessing do not match.")

    def load(self, example: dict, video: dict, config: ModelConfig) -> tuple[np.ndarray, dict]:
        spec = input_spec(example, video, config)
        key = spec_key(spec)
        entry = self.manifest["entries"].get(key)
        if entry is None or entry["spec"] != spec:
            raise ValueError(f"No matching player crops for shot {example['shot']['id']}; rebuild the cache.")
        with np.load(self.folder / f"{key}.npz", allow_pickle=False) as saved:
            frames = saved["frames"]
        if frames.shape != (len(example["indices"]), config.image_size, config.image_size, 3) or frames.dtype != np.uint8:
            raise ValueError(f"Invalid cached player images for shot {example['shot']['id']}.")
        return frames, entry


def parser():
    p = argparse.ArgumentParser(description="Detect annotated hitters and cache upright, pre-contact YOLO crops.")
    p.add_argument("--run", type=Path, default=Path("runs") / "exploratory-121")
    p.add_argument("--output", type=Path, default=Path("runs") / "player-crop-121")
    p.add_argument("--threads", type=int, default=4)
    return p


def run(args) -> Path:
    from .dataset import ShotDataset
    from .explain import input_images, save_image, sequence_sheet

    if args.threads < 1:
        raise ValueError("Threads must be positive.")
    if args.output.exists():
        raise ValueError(f"{args.output} already exists. Choose a new output folder.")
    report = json.loads((args.run / "metrics.json").read_text(encoding="utf-8"))
    project = read_project(args.run / "project_snapshot.json")
    wanted = set(report["train_shot_ids"] + report["validation_shot_ids"])
    shots = [shot for shot in project["shots"] if shot["id"] in wanted]
    if {s["id"] for s in shots} != wanted:
        raise ValueError("The source run snapshot does not contain all training/validation shots.")
    torch.set_num_threads(args.threads)
    cropper = PlayerCropper()
    config = ModelConfig(**{**report["config"], "input_mode": "player_crop", "detector_sha256": cropper.sha256})
    # Build index metadata in full-frame mode; detection consumes raw, pre-contact frames only.
    indexing = ModelConfig(**{**config.to_dict(), "input_mode": "full_frame"})
    data = ShotDataset(project, shots, indexing, report["lead_ms"])
    args.output.mkdir(parents=True)
    folder = args.output / "player_cache"
    folder.mkdir()
    manifest = {
        "version": CROP_VERSION, "detector": DETECTOR_NAME, "detector_sha256": cropper.sha256,
        "config": config.to_dict(), "entries": {},
        "selection": "Court-gated near/far person boxes with within-sequence continuity; no landing/zone labels.",
        "orientation": "Upright for both hitters; player-centered squares with a shared scale plus 60% context.",
    }
    review_rows, failures, overview = [], [], []
    for i, example in enumerate(data.examples):
        video = project["videos"][example["shot"]["video_id"]]
        frames = read_frames(example["path"], example["indices"])
        detections = cropper.detect(frames)
        try:
            crops, info = crop_sequence(
                frames, detections, example["shot"]["hitter"], Court(input_corners(example["shot"], video)),
                config.image_size
            )
        except PlayerNotFoundError as exc:
            failure = {"shot_id": example["shot"]["id"], "reason": str(exc),
                       "indices": example["indices"], "detections": [b.tolist() for b in detections]}
            failures.append(failure)
            print(f"FAILED {failure}", flush=True)
            continue
        spec = input_spec(example, video, config)
        key = spec_key(spec)
        np.savez_compressed(folder / f"{key}.npz", frames=crops)
        manifest["entries"][key] = {
            "spec": spec, "shot_id": example["shot"]["id"], **info,
            "detections": [boxes.tolist() for boxes in detections],
        }
        # Audit every sequence in source coordinates, not only successful predictions.
        original = frames[-1].copy()
        bx1, by1, bx2, by2, _ = info["person_boxes"][-1]
        cv2.rectangle(original, (round(bx1), round(by1)), (round(bx2), round(by2)), (40, 240, 40), 4)
        x1, y1, x2, y2 = info["crop_boxes"][-1]
        cv2.rectangle(original, (x1, y1), (x2, y2), (0, 180, 255), 3)
        row = (f"#{i + 1} {example['shot']['hitter']} | contact {example['shot']['contact_frame']}",
               [cv2.resize(original, (400, 225)), crops[-1]])
        review_rows.append(row)
        overview.append((f"#{i + 1} {example['shot']['hitter']}", crops[-1]))
        print(f"Cropped {i + 1}/{len(data)} sequences.", flush=True)
    manifest["failures"] = failures
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for page, start in enumerate(range(0, len(review_rows), 12), 1):
        save_image(args.output / f"crop_review_{page:02d}.jpg", sequence_sheet(
            review_rows[start:start + 12],
            ["YOLO hitter audit: green = person box; orange = player-centered crop.",
             "Both near and far crops remain upright. No future/bounce images used."],
            ["Source view (distorted for display)", "Actual player input"],
            cell=400,
        ))
    if overview:
        columns, tile = 11, 160
        sheet = np.full((40 + math.ceil(len(overview) / columns) * (tile + 24), columns * tile, 3), 24, np.uint8)
        cv2.putText(sheet, "Every pre-contact player crop (final sampled frame); near/far both upright.",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, .65, (235, 235, 235), 1)
        for i, (name, image) in enumerate(overview):
            x, y = (i % columns) * tile, 40 + (i // columns) * (tile + 24)
            cv2.putText(sheet, name, (x + 5, y + 18), cv2.FONT_HERSHEY_SIMPLEX, .45, (230, 230, 230), 1)
            sheet[y + 24:y + 24 + tile, x:x + tile] = cv2.resize(image, (tile, tile))
        save_image(args.output / "all_player_crops.jpg", sheet)
    (args.output / "project_snapshot.json").write_text(json.dumps(project, indent=2), encoding="utf-8")
    (args.output / "preprocessing.json").write_text(json.dumps({
        "source_run": str(args.run.resolve()), "config": config.to_dict(),
        "train_shot_ids": report["train_shot_ids"], "validation_shot_ids": report["validation_shot_ids"],
        "source_examples": len(data), "successful_examples": len(manifest["entries"]),
        "failed_examples": failures,
    }, indent=2), encoding="utf-8")
    if failures:
        raise ValueError(f"{len(failures)} sequences lack a visible hitter. Review failures before training; no full-frame fallback was used.")
    print(f"Saved {len(manifest['entries'])} player sequences to {folder}", flush=True)
    return folder


def main():
    p = parser()
    try:
        run(p.parse_args())
    except (ValueError, OSError) as exc:
        p.exit(1, f"Player preprocessing failed: {exc}\n")


if __name__ == "__main__":
    main()
