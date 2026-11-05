"""Player detection inside court-view segments (YOLO every few frames, near/far selection per frame)."""

import cv2

from ..court import Court
from ..players import PlayerNotFoundError, select_player

DETECT_STRIDE = 3
DETECT_SECONDS = 0.12  # box resolution the engine was validated at: every 3rd frame of 25 fps Turin footage
BATCH = 8


def stride_for(fps: float) -> int:
    """Frames between YOLO detections: a fixed time step, so 60 fps footage is not sampled 2.4x more than it needs."""
    return max(1, round(DETECT_SECONDS * fps))


def detect_player_samples(path, ranges, corners_at, cropper, stride: int = DETECT_STRIDE, progress=None) -> dict[int, dict]:
    """Return {frame: {"near": box|None, "far": box|None}} for every `stride`-th frame of each (start, end) range."""
    samples: dict[int, dict] = {}
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot open {path}")
        for start, end in ranges:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            batch, prior = [], {}

            def flush():
                detections = cropper.detect([frame for _, frame in batch])
                for (index, _), found in zip(batch, detections):
                    corners = corners_at(index)
                    entry = {"near": None, "far": None}
                    if corners is not None:
                        court = Court([list(map(float, c)) for c in corners])
                        for side in ("near", "far"):
                            try:
                                entry[side] = select_player(found, side, court, prior.get(side))
                                prior[side] = entry[side]
                            except PlayerNotFoundError:
                                prior.pop(side, None)
                    samples[index] = entry
                batch.clear()

            for index in range(start, end + 1):
                ok, frame = cap.read()
                if not ok:
                    break
                if (index - start) % stride == 0:
                    batch.append((index, frame))
                    if len(batch) == BATCH:
                        flush()
                        if progress:
                            progress(index)
            if batch:
                flush()
    finally:
        cap.release()
    return samples


class BoxLookup:
    """Nearest-sample player boxes; a box older than one stride is treated as unavailable."""

    def __init__(self, samples: dict[int, dict], stride: int = DETECT_STRIDE):
        self.samples = samples
        self.frames = sorted(samples)
        self.stride = stride

    def at(self, frame: int, side: str):
        import bisect

        if not self.frames:
            return None
        i = bisect.bisect_left(self.frames, frame)
        options = [self.frames[j] for j in (i - 1, i) if 0 <= j < len(self.frames)]
        nearest = min(options, key=lambda f: abs(f - frame))
        if abs(nearest - frame) > self.stride:
            return None
        return self.samples[nearest].get(side)
