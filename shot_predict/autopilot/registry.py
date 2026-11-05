"""Registering a camera from the labeling app: four clicks on a wide court view make every clip from that camera analyzable."""

from pathlib import Path

import numpy as np

from ..video import read_frame
from . import camera

MAX_NAME_LENGTH = 60


def unique_name(base: str, taken: set[str]) -> str:
    base = "".join(c if c.isalnum() or c in "-_." else "-" for c in base.strip())[:MAX_NAME_LENGTH] or "camera"
    name, counter = base, 2
    while name in taken:
        name, counter = f"{base}-{counter}", counter + 1
    return name


def register_reference(video: dict, frame_index: int, corners: list, library_path: Path, name: str | None = None) -> dict:
    """Check the clicked corners against the frame's court lines, then add the frame to the camera library.

    The same line check later accepts or rejects every frame of every clip, so corners that fail it here would make
    the whole camera analyze as "no court view"; refusing them now saves a wasted batch.
    """
    if not 0 <= frame_index < video["frames"]:
        raise ValueError("That frame is outside the clip.")
    frame = read_frame(video["path"], frame_index)
    height, width = frame.shape[:2]
    array = np.asarray(corners, float)
    if not camera.plausible(array, (width, height)):
        raise ValueError("Those corners do not look like a court seen from the broadcast camera. "
                         "Click near-left, near-right, far-right, then far-left singles corners.")
    score, checked = camera.line_score(frame, array)
    if checked < camera.MIN_LINE_POINTS or score < camera.MIN_LINE_SCORE:
        raise ValueError(
            f"The court lines do not line up with your corners (found {score:.0%} of the lines; need "
            f"{camera.MIN_LINE_SCORE:.0%}). Pick a sharp, wide view where the whole court is visible and click the "
            "singles corners precisely.")
    camera.build_reference("check", frame, array)  # raises when the frame has too few stable features to match later
    entries = camera.read_library_entries(library_path)
    entry_name = unique_name(name or f"{video['match_id']}-main", {e["name"] for e in entries})
    entries.append({
        "name": entry_name, "video_id": None, "path": video["path"], "frame": int(frame_index),
        "corners": [[float(x), float(y)] for x, y in array], "status": "confirmed_by_user",
    })
    camera.write_library_entries(library_path, entries)
    return {"name": entry_name, "line_score": round(score, 3)}
