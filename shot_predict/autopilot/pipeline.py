"""Resumable per-clip analysis: camera -> players -> ball -> proposals. Never reads or writes labels."""

import json
import time
from pathlib import Path

import cv2
import numpy as np

from . import ball, camera, events, inbox, playerpass

PIPELINE_VERSION = "autopilot-1"
MIN_SEGMENT_SECONDS = 2.0


def source_identity(video: dict) -> dict:
    stat = Path(video["path"]).stat()
    return {"path": str(Path(video["path"]).resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    temporary.replace(path)


def _read_valid(path: Path, identity: dict, version: str):
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("source") != identity or data.get("version") != version:
        return None
    return data


def fixed_camera_samples(video: dict, start: int, end: int) -> list[dict]:
    """A steady, user-calibrated clip: the same corners at every sampled frame."""
    step = max(1, round(camera.SCAN_STEP_SECONDS * video["fps"]))
    corners = [list(map(float, p)) for p in video["corners"]]
    return [
        {"frame": f, "accepted": True, "reference": "fixed", "corners": corners, "reason": "ok", "inliers": 0, "line_score": 1.0}
        for f in range(start, end, step)
    ]


def is_complete(folder: Path, video: dict) -> bool:
    """Cheap check (a tiny marker file) that a clip was fully analyzed by the current pipeline and engine."""
    try:
        marker = json.loads((Path(folder) / "done.json").read_text(encoding="utf-8"))
        return (marker.get("version") == PIPELINE_VERSION and marker.get("engine_version") == events.ENGINE_VERSION
                and marker.get("source") == source_identity(video))
    except (OSError, ValueError, KeyError):
        return False


def _mark_done(folder: Path, identity: dict, proposals: list, segments: list, fps: float) -> dict:
    marker = {
        "version": PIPELINE_VERSION, "engine_version": events.ENGINE_VERSION, "source": identity,
        "proposals": len(proposals), "validatable": sum(inbox.is_validatable(p, fps) for p in proposals),
        "segments": len(segments), "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    _write(folder / "done.json", marker)
    return marker


def analyze_clip(project, video_id, analysis_root: Path, references, cropper, frame_range=None, log=print) -> dict:
    video = project["videos"][video_id]
    folder = analysis_root / video_id
    identity = source_identity(video)
    fps, frames = video["fps"], video["frames"]
    start_limit, end_limit = frame_range or (0, frames)
    label = f"{Path(video['path']).name}"
    t0 = time.time()

    # 1. Camera
    camera_path = folder / "camera.json"
    cached = json.loads(camera_path.read_text(encoding="utf-8")) if camera_path.is_file() else None
    if cached and cached.get("source") == identity and cached.get("range") == [start_limit, end_limit]:
        track = camera.CameraTrack.from_dict(cached)
        log(f"[{label}] camera: reused {len(track.segments)} segments")
    else:
        if video.get("corners"):
            samples = fixed_camera_samples(video, start_limit, end_limit)
        else:
            if not references:
                raise ValueError("This clip has no fixed calibration; add a camera reference first.")
            log(f"[{label}] camera: scanning (about 1.5 minutes per 5-minute clip)...")
            samples = camera.scan_clip(video["path"], references, fps, end_limit)
            samples = [s for s in samples if s["frame"] >= start_limit]
        track = camera.CameraTrack(samples, fps)
        _write(camera_path, {**track.to_dict(), "source": identity, "range": [start_limit, end_limit], "version": PIPELINE_VERSION})
        log(f"[{label}] camera: {track.coverage(frames)} ({time.time() - t0:.0f}s)")
    segments = [
        [s["start"], s["end"]] for s in track.segments if (s["end"] - s["start"]) >= MIN_SEGMENT_SECONDS * fps
    ]
    if not segments:
        result = {"version": PIPELINE_VERSION, "source": identity, "video_id": video_id, "fps": fps, "frames": frames,
                  "engine_version": events.ENGINE_VERSION, "range": [start_limit, end_limit], "proposals": [], "segments": []}
        _write(folder / "proposals.json", result)
        inbox.write_inbox(folder)
        _mark_done(folder, identity, [], [], fps)
        log(f"[{label}] no court-view segments of {MIN_SEGMENT_SECONDS:.0f}s or more")
        return result

    # 2. Players
    players_path = folder / "players.json"
    stride = playerpass.stride_for(fps)
    cached = _read_valid(players_path, identity, PIPELINE_VERSION)
    # Samples taken at a finer stride than needed are fine to reuse; a coarser cache is redone.
    if cached and cached.get("segments") == segments and cached.get("stride", playerpass.DETECT_STRIDE) <= stride:
        samples = {int(k): v for k, v in cached["samples"].items()}
        log(f"[{label}] players: reused {len(samples)} samples")
    else:
        log(f"[{label}] players: YOLO on {sum(e - s for s, e in segments) / fps:.0f}s of court view (every {stride} frames)...")
        samples = playerpass.detect_player_samples(
            video["path"], segments, track.corners_at, cropper, stride=stride,
            progress=lambda f: log(f"[{label}] players through frame {f}"),
        )
        _write(players_path, {"version": PIPELINE_VERSION, "source": identity, "segments": segments, "stride": stride,
                              "samples": {str(k): {s: (None if b is None else [float(v) for v in b]) for s, b in v.items()}
                                          for k, v in samples.items()}})
        log(f"[{label}] players: {len(samples)} samples ({time.time() - t0:.0f}s)")
    lookup = playerpass.BoxLookup(samples, stride=stride)

    # 3. Ball
    ball_path = folder / "ball.json"
    cached = _read_valid(ball_path, identity, PIPELINE_VERSION)
    if cached and cached.get("segments") == segments:
        tracklets = cached["tracklets"]
        log(f"[{label}] ball: reused {sum(len(t) for t in tracklets)} tracklets")
    else:
        roi_cache = {}

        def roi_for(frame):
            corners = track.corners_at(frame)
            if corners is None:
                return None
            key = tuple(np.round(corners, 0).ravel())
            if key not in roi_cache:
                roi_cache[key] = ball.court_roi(corners, (video["width"], video["height"]))
            return roi_cache[key]

        tracklets = []
        for number, (start, end) in enumerate(segments):
            found = ball.scan_candidates(
                video["path"], start + 1, end, roi_for,
                lambda f: [ball.expand_box(lookup.at(f, "near")), ball.expand_box(lookup.at(f, "far"))],
            )
            found = ball.drop_persistent_noise(found, (video["width"], video["height"]))
            tracklets.append(ball.link_tracklets(found, fps))
            log(f"[{label}] ball: segment {number + 1}/{len(segments)} -> {len(tracklets[-1])} tracklets")
        _write(ball_path, {"version": PIPELINE_VERSION, "source": identity, "segments": segments, "tracklets": tracklets})
        log(f"[{label}] ball: done ({time.time() - t0:.0f}s)")

    # 4. Proposals
    proposals = []
    for (start, end), tracks in zip(segments, tracklets):
        proposals.extend(events.propose_shots(tracks, fps, track.corners_at, lookup.at, (start, end), video_id))
    result = {
        "version": PIPELINE_VERSION, "engine_version": events.ENGINE_VERSION, "source": identity, "video_id": video_id,
        "fps": fps, "frames": frames, "range": [start_limit, end_limit], "segments": segments, "proposals": proposals,
    }
    _write(folder / "proposals.json", result)
    inbox.write_inbox(folder)
    if frame_range is None:  # a partial-range analysis is an experiment, not a finished clip
        _mark_done(folder, identity, proposals, segments, fps)
    log(f"[{label}] proposals: {len(proposals)} ({time.time() - t0:.0f}s total)")
    return result
