"""Match manifests: how footage reaches the batch analysis and the labeling app without anyone editing project.json.

A manifest (`data/matches/<match_id>.json`) names a folder of clips. The batch runner analyzes those clips; the app
imports them on startup (it is the only writer of project.json, so nothing here ever writes the project file except
through `sync_library`, which the app calls). A full-length recording is first split into clips by `split_match`.
"""

import csv
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ..storage import add_video, video_id_for

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv")
MANIFEST_VERSION = 1
DEFAULT_CLIP_SECONDS = 300
MATCH_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def check_match_id(match_id: str) -> str:
    match_id = match_id.strip()
    if not MATCH_ID_PATTERN.fullmatch(match_id):
        raise ValueError("Match IDs may use letters, digits, dot, dash and underscore (for example usopen_2026_r4_alcaraz_paul).")
    return match_id


def clips_in(folder: Path) -> list[Path]:
    return sorted(p for p in Path(folder).iterdir() if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES)


def manifest_path(matches_dir: Path, match_id: str) -> Path:
    return Path(matches_dir) / f"{check_match_id(match_id)}.json"


def write_manifest(matches_dir: Path, match_id: str, clips_folder: Path, near_player: str = "", far_player: str = "",
                   source: str | None = None) -> Path:
    clips_folder = Path(clips_folder).expanduser().resolve(strict=True)
    if not clips_in(clips_folder):
        raise ValueError(f"No video clips found in {clips_folder}.")
    path = manifest_path(matches_dir, match_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "version": MANIFEST_VERSION, "match_id": check_match_id(match_id), "clips_folder": str(clips_folder),
        "near_player": near_player.strip(), "far_player": far_player.strip(), "source": source, "ready": True,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def read_manifests(matches_dir: Path) -> list[dict]:
    """Ready, readable manifests whose clip folder still exists; anything else is ignored."""
    manifests = []
    for path in sorted(Path(matches_dir).glob("*.json")) if Path(matches_dir).is_dir() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if data.get("version") == MANIFEST_VERSION and data.get("ready") and Path(data["clips_folder"]).is_dir():
                data["match_id"] = check_match_id(data["match_id"])
                manifests.append(data)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return manifests


def library_stamp(matches_dir: Path) -> tuple:
    """Changes whenever a manifest is added or rewritten; lets the app skip the sync on most reruns."""
    folder = Path(matches_dir)
    if not folder.is_dir():
        return ()
    return tuple((p.name, p.stat().st_mtime_ns) for p in sorted(folder.glob("*.json")))


def sync_library(project: dict, matches_dir: Path) -> list[str]:
    """Import every clip of every ready manifest that the project does not have yet; returns the new video IDs."""
    added = []
    for manifest in read_manifests(matches_dir):
        for clip in clips_in(Path(manifest["clips_folder"])):
            if video_id_for(clip) in project["videos"]:
                continue
            try:
                added.append(add_video(project, clip, manifest["match_id"], manifest.get("near_player", ""),
                                       manifest.get("far_player", "")))
            except (ValueError, OSError):
                continue  # an unreadable clip must not block the rest of the library
    return added


def ffmpeg_executable() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def split_match(source: Path, clips_folder: Path, clip_seconds: int = DEFAULT_CLIP_SECONDS, reencode: bool = False,
                ffmpeg: str | None = None) -> list[Path]:
    """Cut a long recording into clip_000.mp4, clip_001.mp4, ... plus clip_times.csv (name,start,end in seconds).

    Stream copy by default: fast and lossless, cutting at the nearest keyframes, so clips are close to
    `clip_seconds` rather than exact. `reencode` makes constant-frame-rate H.264 for sources the browser cannot play.
    """
    source = Path(source).expanduser().resolve(strict=True)
    clips_folder = Path(clips_folder)
    if clips_folder.exists() and any(clips_folder.iterdir()):
        raise ValueError(f"{clips_folder} already has files; choose another match ID or remove it first.")
    clips_folder.mkdir(parents=True, exist_ok=True)
    codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac", "-force_key_frames", f"expr:gte(t,n_forced*{clip_seconds})"] \
        if reencode else ["-c", "copy"]
    command = [
        ffmpeg or ffmpeg_executable(), "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
        "-map", "0:v:0", "-map", "0:a:0?", *codec, "-f", "segment", "-segment_time", str(clip_seconds),
        "-reset_timestamps", "1", "-segment_format", "mp4", "-segment_format_options", "movflags=+faststart",
        "-segment_list", str(clips_folder / "clip_times.csv"), "-segment_list_type", "csv",
        str(clips_folder / "clip_%03d.mp4"),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    clips = clips_in(clips_folder)
    if result.returncode != 0 or not clips:
        raise ValueError(f"Splitting failed: {result.stderr.strip()[-400:] or 'no clips were produced'}. "
                         "If the source codec cannot go into MP4 as-is, retry with --reencode.")
    return clips


def clip_times(clips_folder: Path) -> dict[str, tuple[float, float]]:
    path = Path(clips_folder) / "clip_times.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row[0]: (float(row[1]), float(row[2])) for row in csv.reader(handle) if len(row) >= 3}


def register_match(source: Path, match_id: str, matches_dir: Path, videos_root: Path, near_player: str = "",
                   far_player: str = "", clip_seconds: int = DEFAULT_CLIP_SECONDS, reencode: bool = False) -> dict:
    """Make a match known: split a recording file into clips, or adopt an existing folder of clips in place."""
    match_id = check_match_id(match_id)
    source = Path(source).expanduser().resolve(strict=True)
    if source.is_dir():
        clips_folder, split = source, False
    else:
        clips_folder = Path(videos_root) / match_id
        split_match(source, clips_folder, clip_seconds, reencode)
        split = True
    path = write_manifest(matches_dir, match_id, clips_folder, near_player, far_player, str(source))
    return {"match_id": match_id, "manifest": str(path), "clips_folder": str(clips_folder),
            "clips": len(clips_in(clips_folder)), "split": split}
