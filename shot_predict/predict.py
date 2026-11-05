import argparse
import csv
from collections import deque
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
import torch

from .court import Court, ZONES
from .model import ModelConfig, ShotModel
from .players import PlayerCropper, PlayerNotFoundError, crop_sequence
from .storage import read_project
from .train import choose_device
from .video import metadata, prepare_frame


def load_model(checkpoint: Path, device: torch.device):
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if saved.get("version") != 1 or saved.get("zones") != list(ZONES):
        raise ValueError("Unsupported checkpoint or zone order.")
    config = ModelConfig(**saved["config"])
    model = ShotModel(config).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return model, config, saved["lead_ms"]


def draw_prediction(frame, court, hitter, probabilities, threshold, row):
    best = int(np.argmax(probabilities))
    confidence = float(probabilities[best])
    color = (255, 180, 40) if hitter == "near" else (80, 210, 255)
    label = f"{hitter}: {ZONES[best].replace('_', ' ')} {confidence:.0%} (raw)"
    cv2.putText(frame, label, (15, 55 + row * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
    if confidence >= threshold:
        # Initial prototype uses a fixed origin, not an automatically tracked player position.
        start = court.pixel(0.5, 0.95 if hitter == "near" else 0.05)
        end = court.target(ZONES[best], hitter)
        overlay = frame.copy()
        cv2.arrowedLine(overlay, start, end, color, 4, tipLength=0.06)
        cv2.circle(overlay, end, 9, color, 2)
        cv2.addWeighted(overlay, confidence, frame, 1 - confidence, 0, dst=frame)


def parser():
    p = argparse.ArgumentParser(description="Export rolling tennis predictions to an MP4 plus probability CSV.")
    p.add_argument("--project", type=Path, default=Path("data") / "project.json")
    p.add_argument("--video-id", required=True, help="Imported/calibrated video ID in project.json.")
    p.add_argument("--checkpoint", type=Path, default=Path("runs") / "baseline" / "best.pt")
    p.add_argument("--output", type=Path, default=Path("runs") / "prediction.mp4")
    p.add_argument("--hitter", choices=["near", "far", "both"], default="both")
    p.add_argument("--stride", type=int, default=3, help="Update every N video frames.")
    p.add_argument("--threshold", type=float, default=0.45, help="Minimum raw probability to draw a line.")
    p.add_argument("--smoothing", type=float, default=0.5, help="EMA weight on newest prediction (0 < value <= 1).")
    p.add_argument("--device", default="auto")
    p.add_argument("--threads", type=int, default=4)
    return p


def run(args) -> tuple[Path, Path]:
    if args.stride < 1 or args.threads < 1 or not 0 <= args.threshold <= 1 or not 0 < args.smoothing <= 1:
        raise ValueError("Invalid stride, threads, probability threshold, or smoothing.")
    if args.output.suffix.lower() != ".mp4":
        raise ValueError("Output must have an .mp4 extension.")
    project = read_project(args.project)
    if args.video_id not in project["videos"]:
        raise ValueError("Unknown video ID. Import and calibrate the video in the app first.")
    video = project["videos"][args.video_id]
    if not video["corners"]:
        raise ValueError("Calibrate this video in the labeling app first.")
    source = Path(video["path"])
    if args.output.resolve() == source.resolve():
        raise ValueError("Output must not overwrite the source video.")
    csv_path = args.output.with_suffix(".csv")
    if args.output.exists() or csv_path.exists():
        raise ValueError("Output or probability CSV already exists. Choose a new --output.")
    info = metadata(source)
    court = Court(video["corners"])
    torch.set_num_threads(args.threads)
    device = choose_device(args.device)
    model, config, leads = load_model(args.checkpoint, device)
    cropper = PlayerCropper(config.detector_sha256) if config.input_mode == "player_crop" else None
    duration_frames = max(1, round(config.seconds * info["fps"]))
    if info["frames"] <= duration_frames + args.stride:
        raise ValueError("Video is too short to collect the model's context and produce predictions.")
    buffer = deque(maxlen=duration_frames + 1)
    hitters = ["near", "far"] if args.hitter == "both" else [args.hitter]
    probabilities = {}
    unavailable = {}
    prediction_count = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    temporary_video = args.output.with_name(f"{args.output.stem}.{token}.tmp.mp4")
    temporary_csv = csv_path.with_name(f"{csv_path.stem}.{token}.tmp.csv")
    cap = cv2.VideoCapture(str(source))
    writer = cv2.VideoWriter(
        str(temporary_video), cv2.VideoWriter_fourcc(*"mp4v"), info["fps"], (info["width"], info["height"])
    )
    if not writer.isOpened():
        cap.release()
        writer.release()
        temporary_video.unlink(missing_ok=True)
        raise ValueError("MP4 encoder unavailable in this OpenCV installation.")
    print(f"Device: {device}; training lead times: {leads} ms. "
          "Rolling forecasts are not contact/turn-gated; probabilities are uncalibrated.", flush=True)
    count = 0
    try:
        with temporary_csv.open("w", newline="", encoding="utf-8") as handle:
            csv_writer = csv.writer(handle)
            csv_writer.writerow(["frame", "seconds", "hitter", *(["status"] if cropper else []), *ZONES])
            with torch.inference_mode():
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        if count < info["frames"]:
                            raise ValueError(f"Video decoding ended early at frame {count} of {info['frames']}.")
                        break
                    buffer.append(
                        frame.copy() if cropper else cv2.resize(frame, (config.image_size, config.image_size))
                    )
                    if len(buffer) == buffer.maxlen and count % args.stride == 0:
                        frames = list(buffer)
                        indices = np.linspace(0, len(frames) - 1, config.samples).round().astype(int)
                        selected = [frames[i] for i in indices]
                        detections = cropper.detect(selected) if cropper else []
                        clips, visible = [], []
                        for hitter in hitters:
                            if cropper:
                                try:
                                    crops, _ = crop_sequence(
                                        selected, detections, hitter, court, config.image_size
                                    )
                                except PlayerNotFoundError as exc:
                                    probabilities.pop(hitter, None)
                                    reason = str(exc)
                                    if unavailable.get(hitter) != reason:
                                        print(f"Frame {count}: {reason}", flush=True)
                                    unavailable[hitter] = reason
                                    csv_writer.writerow([count, count / info["fps"], hitter, reason, *([""] * len(ZONES))])
                                    continue
                                images = [prepare_frame(crop, config.image_size, "near") for crop in crops]
                            else:
                                images = [prepare_frame(image, config.image_size, hitter) for image in selected]
                            unavailable.pop(hitter, None)
                            clips.append(np.stack(images))
                            visible.append(hitter)
                        if clips:
                            predicted = model(torch.from_numpy(np.stack(clips)).to(device)).softmax(dim=1).cpu().numpy()
                            if not np.isfinite(predicted).all():
                                raise ValueError("Model produced non-finite probabilities.")
                            for hitter, current in zip(visible, predicted):
                                if hitter in probabilities:
                                    current = args.smoothing * current + (1 - args.smoothing) * probabilities[hitter]
                                probabilities[hitter] = current
                                prediction_count += 1
                                csv_writer.writerow([count, count / info["fps"], hitter,
                                                     *(["ok"] if cropper else []), *current.tolist()])
                    output = frame.copy()
                    caption = "EXPERIMENTAL - YOLO crops / no turn detection" if cropper else "EXPERIMENTAL - fixed origins / no turn detection"
                    cv2.putText(output, caption, (15, 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
                    if not probabilities:
                        cv2.putText(output, "Collecting context...", (15, 55),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
                    for row, hitter in enumerate(hitters):
                        if hitter in probabilities:
                            draw_prediction(output, court, hitter, probabilities[hitter], args.threshold, row)
                        elif hitter in unavailable:
                            cv2.putText(output, f"{hitter}: player not visible / no forecast", (15, 55 + row * 28),
                                        cv2.FONT_HERSHEY_SIMPLEX, .6, (80, 120, 255), 2)
                    writer.write(output)
                    count += 1
        writer.release()
        exported = metadata(temporary_video)
        if exported["frames"] != count or count == 0 or prediction_count == 0:
            raise ValueError("Export verification failed: frame count mismatch or no predictions.")
        temporary_video.replace(args.output)
        temporary_csv.replace(csv_path)
    finally:
        cap.release()
        writer.release()
        temporary_video.unlink(missing_ok=True)
        temporary_csv.unlink(missing_ok=True)
    print(f"Exported {count} frames to {args.output}; probabilities: {csv_path}. Audio is not retained.", flush=True)
    return args.output, csv_path


def main():
    p = parser()
    try:
        run(p.parse_args())
    except (ValueError, OSError) as exc:
        p.exit(1, f"Export failed: {exc}\n")


if __name__ == "__main__":
    main()
