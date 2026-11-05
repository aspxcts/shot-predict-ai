from pathlib import Path

import cv2
import numpy as np


def metadata(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        result = {
            "fps": float(cap.get(cv2.CAP_PROP_FPS)),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
        if result["fps"] <= 0 or result["frames"] < 2 or min(result["width"], result["height"]) <= 0:
            raise ValueError("Video has invalid frame count, dimensions, or frame rate.")
        return result
    finally:
        cap.release()


def read_frame(path: str | Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened() or index < 0:
            raise ValueError(f"Cannot read frame {index} from {path}")
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            raise ValueError(f"Cannot decode frame {index} from {path}")
        return frame
    finally:
        cap.release()


def clip_indices(contact: int, fps: float, seconds: float, samples: int, lead_ms: float) -> list[int]:
    if fps <= 0 or seconds <= 0 or samples < 2 or lead_ms < 0:
        raise ValueError("FPS/window must be positive, samples >= 2, and lead >= 0.")
    # Contact and all post-contact frames are excluded, even with zero lead.
    end = contact - max(1, int(np.ceil(lead_ms * fps / 1000)))
    start = end - max(1, round(seconds * fps))
    if start < 0:
        raise ValueError("Not enough pre-contact context; choose a later shot or shorter window.")
    return np.linspace(start, end, samples).round().astype(int).tolist()


def prepare_frame(frame: np.ndarray, size: int, hitter: str) -> np.ndarray:
    if hitter not in ("near", "far"):
        raise ValueError("Hitter must be near or far.")
    if hitter == "far":
        frame = cv2.rotate(frame, cv2.ROTATE_180)
    rgb = cv2.cvtColor(cv2.resize(frame, (size, size)), cv2.COLOR_BGR2RGB)
    image = rgb.astype(np.float32) / 255
    image = (image - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    return np.transpose(image, (2, 0, 1))


def read_frames(path: str | Path, indices: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {path}")
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f"Cannot decode training frame {index} from {path}")
            frames.append(frame)
    finally:
        cap.release()
    return frames


def read_clip(path: str | Path, indices: list[int], size: int, hitter: str) -> np.ndarray:
    return np.stack([prepare_frame(frame, size, hitter) for frame in read_frames(path, indices)])
