"""Unattended analysis of many clips with a pool of worker processes.

Safe to stop and restart: finished clips are skipped (tiny `done.json` markers), interrupted clips resume at their last
finished stage. Workers only write analysis sidecars under the analysis folder; they never touch project.json or labels,
so the labeling app can run (and clips can be reviewed) while the batch is still going.
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ..storage import read_project, video_id_for
from ..video import metadata
from . import camera, inbox, matches as match_library, pipeline

LEDGER_NAME = "batch_status.json"
BELOW_NORMAL_PRIORITY = 0x00004000


def lower_priority() -> None:
    """Keep the machine responsive for whoever is labeling while the batch runs (Windows; no-op elsewhere)."""
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE  # a 64-bit pseudo-handle: the default int type truncates it
        kernel32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel32.SetPriorityClass.restype = wintypes.BOOL
        kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), BELOW_NORMAL_PRIORITY)
    except (AttributeError, OSError, ImportError):
        pass


def collect_tasks(project: dict, matches_dir: Path, analysis_root: Path, match_id: str | None = None,
                  limit: int | None = None) -> tuple[list[dict], dict]:
    """Clips that still need analysis (project clips and manifest clips), plus a count of what was skipped."""
    tasks, seen, skipped = [], set(), 0

    def wanted(video_id: str, video: dict) -> bool:
        nonlocal skipped
        if video_id in seen:
            return False
        seen.add(video_id)
        if pipeline.is_complete(Path(analysis_root) / video_id, video):
            skipped += 1
            return False
        return True

    ordered = sorted(project["videos"].items(), key=lambda kv: (kv[1]["match_id"], kv[1]["path"]))
    for video_id, video in ordered:
        if match_id in (None, video["match_id"]) and wanted(video_id, video):
            tasks.append({"video_id": video_id, "video": video})
    for manifest in match_library.read_manifests(matches_dir):
        if match_id not in (None, manifest["match_id"]):
            continue
        for clip in match_library.clips_in(Path(manifest["clips_folder"])):
            video_id = video_id_for(clip)
            if video_id in seen or video_id in project["videos"]:
                continue
            if not wanted(video_id, {"path": str(clip.resolve())}):
                continue
            try:
                video = {**metadata(clip), "path": str(clip.resolve()), "match_id": manifest["match_id"],
                         "near_player": manifest.get("near_player", ""), "far_player": manifest.get("far_player", ""),
                         "corners": None}
            except ValueError:
                continue
            tasks.append({"video_id": video_id, "video": video})
    return (tasks[:limit] if limit else tasks), {"already_done": skipped}


_WORKER: dict = {}


def _init_worker(analysis_root: str, library_path: str, reference_videos: dict, threads: int) -> None:
    lower_priority()
    import cv2
    import torch

    cv2.setNumThreads(threads)
    torch.set_num_threads(threads)
    from ..players import PlayerCropper

    entries = camera.read_library_entries(Path(library_path))
    _WORKER["analysis_root"] = Path(analysis_root)
    _WORKER["cropper"] = PlayerCropper()
    _WORKER["references"] = camera.load_references(entries, {"videos": reference_videos}) if entries else []


def _analyze_task(task: dict) -> dict:
    video_id, video = task["video_id"], task["video"]
    started = time.time()
    try:
        result = pipeline.analyze_clip(
            {"videos": {video_id: video}}, video_id, _WORKER["analysis_root"], _WORKER["references"],
            _WORKER["cropper"], None, log=lambda message: None,
        )
        fps = video["fps"]
        return {
            "video_id": video_id, "status": "done" if result["segments"] else "no_court_view",
            "seconds": round(time.time() - started, 1), "proposals": len(result["proposals"]),
            "segments": len(result["segments"]),
            "validatable": sum(inbox.is_validatable(p, fps) for p in result["proposals"]),
        }
    except Exception as exc:  # one bad clip must never stop the batch
        message = f"{type(exc).__name__}: {exc}"
        return {"video_id": video_id, "status": "needs_reference" if "camera reference" in message else "failed",
                "seconds": round(time.time() - started, 1), "error": message[:300]}


def read_ledger(analysis_root: Path) -> dict:
    try:
        return json.loads((Path(analysis_root) / LEDGER_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"clips": {}}


def write_ledger(analysis_root: Path, ledger: dict) -> None:
    path = Path(analysis_root) / LEDGER_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    ledger["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    temporary.write_text(json.dumps(ledger, indent=1), encoding="utf-8")
    temporary.replace(path)


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m {seconds % 60:02d}s"


def status_report(project: dict, matches_dir: Path, analysis_root: Path) -> list[dict]:
    """Per-match progress from the tiny done markers and the batch ledger (never parses proposal files)."""
    clips: dict[str, dict] = {}
    for video_id, video in project["videos"].items():
        clips[video_id] = {"match_id": video["match_id"], "name": Path(video["path"]).name, "video": video}
    for manifest in match_library.read_manifests(matches_dir):
        for clip in match_library.clips_in(Path(manifest["clips_folder"])):
            clips.setdefault(video_id_for(clip), {"match_id": manifest["match_id"], "name": clip.name,
                                                  "video": {"path": str(clip.resolve())}})
    ledger = read_ledger(analysis_root)["clips"]
    report: dict[str, dict] = {}
    for video_id, clip in clips.items():
        row = report.setdefault(clip["match_id"], {
            "match_id": clip["match_id"], "clips": 0, "analyzed": 0, "proposals": 0, "cards": 0,
            "needs_reference": 0, "no_court_view": 0, "failed": 0})
        row["clips"] += 1
        folder = Path(analysis_root) / video_id
        if pipeline.is_complete(folder, clip["video"]):
            marker = json.loads((folder / "done.json").read_text(encoding="utf-8"))
            row["analyzed"] += 1
            row["proposals"] += marker.get("proposals", 0)
            row["cards"] += marker.get("validatable", 0)
            row["no_court_view"] += 0 if marker.get("segments") else 1
        else:
            status = ledger.get(video_id, {}).get("status")
            if status in ("needs_reference", "failed"):
                row[status] += 1
    return sorted(report.values(), key=lambda r: r["match_id"])


def run_batch(project_path: Path, matches_dir: Path, analysis_root: Path, library_path: Path, workers: int = 3,
              threads: int = 2, match_id: str | None = None, limit: int | None = None, log=print) -> dict:
    project = read_project(project_path)  # read only: the app is the sole writer of this file
    tasks, info = collect_tasks(project, matches_dir, analysis_root, match_id, limit)
    summary = {"queued": len(tasks), **info, "done": 0, "no_court_view": 0, "needs_reference": 0, "failed": 0}
    if not tasks:
        log(f"Nothing to analyze ({info['already_done']} clips already done).")
        return summary
    entries = camera.read_library_entries(Path(library_path))
    reference_videos = {
        e["video_id"]: {"path": project["videos"][e["video_id"]]["path"]}
        for e in entries if not e.get("path") and e["video_id"] in project["videos"]
    }
    ledger = read_ledger(analysis_root)
    for task in tasks:
        ledger["clips"][task["video_id"]] = {
            "name": Path(task["video"]["path"]).name, "match_id": task["video"]["match_id"], "status": "queued"}
    write_ledger(analysis_root, ledger)
    log(f"Analyzing {len(tasks)} clips with {workers} workers ({threads} threads each); "
        f"{info['already_done']} already done. Safe to stop with Ctrl+C and rerun.")
    started, finished = time.time(), 0
    pool = ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                               initargs=(str(analysis_root), str(library_path), reference_videos, threads))
    try:
        futures = {pool.submit(_analyze_task, task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as exc:  # a crashed worker process
                result = {"video_id": task["video_id"], "status": "failed", "seconds": 0, "error": f"{type(exc).__name__}: {exc}"}
            finished += 1
            summary[result["status"]] = summary.get(result["status"], 0) + 1
            ledger["clips"][result["video_id"]].update({**result, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
            write_ledger(analysis_root, ledger)
            rate = (time.time() - started) / finished
            remaining = (len(tasks) - finished) * rate
            detail = (f"{result.get('proposals', 0)} proposals, {result.get('validatable', 0)} cards"
                      if result["status"] in ("done", "no_court_view") else result.get("error", ""))
            log(f"[{finished}/{len(tasks)}] {ledger['clips'][result['video_id']]['name']} {result['status']} "
                f"in {_duration(result['seconds'])}: {detail} | about {_duration(remaining)} left")
    except KeyboardInterrupt:
        pool.shutdown(wait=False, cancel_futures=True)
        log("Stopped. Finished clips are kept; run the same command to continue.")
        raise
    else:
        pool.shutdown()
    return summary
