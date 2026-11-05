import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from shot_predict.autopilot import batch, matches, pipeline
from shot_predict.storage import add_video, video_id_for
from shot_predict.video import metadata


def write_video(path: Path, seconds: int = 30, fps: int = 20) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    assert writer.isOpened()
    for index in range(seconds * fps):
        image = np.full((120, 160, 3), (index * 5) % 255, np.uint8)
        cv2.putText(image, str(index), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        writer.write(image)
    writer.release()
    return path


@pytest.fixture(scope="module")
def recording(tmp_path_factory):
    return write_video(tmp_path_factory.mktemp("source") / "full_match.mp4")


def test_a_recording_is_split_into_playable_clips_that_cover_all_of_it(recording, tmp_path):
    folder = tmp_path / "clips"
    clips = matches.split_match(recording, folder, clip_seconds=10)
    assert 3 <= len(clips) <= 5 and [c.name for c in clips][:2] == ["clip_000.mp4", "clip_001.mp4"]
    total = sum(metadata(c)["frames"] for c in clips)
    assert abs(total - metadata(recording)["frames"]) <= 3
    assert all(metadata(c)["fps"] == pytest.approx(20) for c in clips)
    times = matches.clip_times(folder)
    assert list(times) == [c.name for c in clips] and times["clip_000.mp4"][0] == 0
    assert all(times[c.name][1] > times[c.name][0] for c in clips)
    with pytest.raises(ValueError, match="already has files"):
        matches.split_match(recording, folder, clip_seconds=10)


def test_splitting_something_that_is_not_a_video_fails_cleanly(tmp_path):
    bogus = tmp_path / "not_video.mp4"
    bogus.write_bytes(b"this is not a video")
    with pytest.raises(ValueError, match="Splitting failed"):
        matches.split_match(bogus, tmp_path / "out", clip_seconds=10)


def test_match_ids_are_filename_safe():
    assert matches.check_match_id(" usopen_2026-r4.alcaraz ") == "usopen_2026-r4.alcaraz"
    for bad in ("", "../escape", "a/b", "with space", "-leading-dash", "semi;colon"):
        with pytest.raises(ValueError):
            matches.check_match_id(bad)


def test_registering_a_recording_or_an_existing_folder_writes_a_manifest(recording, tmp_path):
    matches_dir, videos_root = tmp_path / "matches", tmp_path / "videos"
    split = matches.register_match(recording, "match_a", matches_dir, videos_root, "Near N", "Far F", clip_seconds=10)
    assert split["split"] and split["clips"] >= 3 and (videos_root / "match_a" / "clip_000.mp4").is_file()
    adopted_folder = tmp_path / "already_chopped"
    adopted_folder.mkdir()
    write_video(adopted_folder / "one.mp4", seconds=4)
    adopted = matches.register_match(adopted_folder, "match_b", matches_dir, videos_root)
    assert not adopted["split"] and adopted["clips"] == 1 and adopted["clips_folder"] == str(adopted_folder.resolve())
    manifests = {m["match_id"]: m for m in matches.read_manifests(matches_dir)}
    assert set(manifests) == {"match_a", "match_b"} and manifests["match_a"]["near_player"] == "Near N"


def test_sync_imports_new_clips_once_and_ignores_unready_unreadable_or_missing_manifests(recording, tmp_path):
    matches_dir, folder = tmp_path / "matches", tmp_path / "clips"
    folder.mkdir()
    write_video(folder / "a.mp4", seconds=3)
    write_video(folder / "b.mp4", seconds=3)
    (folder / "broken.mp4").write_bytes(b"nope")
    matches.write_manifest(matches_dir, "m1", folder, "N", "F")
    (matches_dir / "unready.json").write_text(json.dumps({"version": 1, "match_id": "u", "clips_folder": str(folder), "ready": False}))
    (matches_dir / "garbage.json").write_text("{not json")
    (matches_dir / "gone.json").write_text(json.dumps({"version": 1, "match_id": "g", "clips_folder": str(tmp_path / "nowhere"), "ready": True}))
    project = {"version": 1, "videos": {}, "shots": []}
    stamp = matches.library_stamp(matches_dir)
    added = matches.sync_library(project, matches_dir)
    assert len(added) == 2 and {v["match_id"] for v in project["videos"].values()} == {"m1"}
    assert {v["near_player"] for v in project["videos"].values()} == {"N"}
    assert matches.sync_library(project, matches_dir) == []  # idempotent
    matches.write_manifest(matches_dir, "m2", folder)
    assert matches.library_stamp(matches_dir) != stamp
    assert matches.library_stamp(tmp_path / "missing") == ()


def make_clip_project(tmp_path):
    folder = tmp_path / "project_clips"
    folder.mkdir()
    project = {"version": 1, "videos": {}, "shots": []}
    ids = [add_video(project, write_video(folder / f"p{i}.mp4", seconds=3), "proj_match") for i in range(2)]
    return project, ids


def test_pending_work_skips_finished_clips_and_includes_unimported_manifest_clips(tmp_path):
    project, ids = make_clip_project(tmp_path)
    analysis, matches_dir = tmp_path / "analysis", tmp_path / "matches"
    folder = tmp_path / "new_clips"
    folder.mkdir()
    new_paths = [write_video(folder / f"n{i}.mp4", seconds=3) for i in range(2)]
    matches.write_manifest(matches_dir, "new_match", folder, "N", "F")
    tasks, info = batch.collect_tasks(project, matches_dir, analysis)
    assert [t["video_id"] for t in tasks][:2] == sorted(ids, key=lambda i: project["videos"][i]["path"])
    assert len(tasks) == 4 and info["already_done"] == 0
    manifest_task = next(t for t in tasks if t["video_id"] == video_id_for(new_paths[0]))
    assert manifest_task["video"]["match_id"] == "new_match" and manifest_task["video"]["corners"] is None
    assert manifest_task["video"]["fps"] == pytest.approx(20)
    # A clip with a current done marker is skipped; one with a stale marker is not.
    done_video = project["videos"][ids[0]]
    pipeline._mark_done(analysis / ids[0], pipeline.source_identity(done_video), [], [], 20.0)
    stale = analysis / ids[1]
    stale.mkdir(parents=True)
    (stale / "done.json").write_text(json.dumps({"version": "old", "engine_version": "old", "source": {}}))
    again, info = batch.collect_tasks(project, matches_dir, analysis)
    assert ids[0] not in [t["video_id"] for t in again] and ids[1] in [t["video_id"] for t in again] and info["already_done"] == 1
    only, _ = batch.collect_tasks(project, matches_dir, analysis, match_id="new_match")
    assert len(only) == 2 and all(t["video"]["match_id"] == "new_match" for t in only)
    assert len(batch.collect_tasks(project, matches_dir, analysis, limit=1)[0]) == 1


def test_a_worker_reports_each_outcome_without_ever_raising(tmp_path, monkeypatch):
    project, ids = make_clip_project(tmp_path)
    video = project["videos"][ids[0]]
    monkeypatch.setattr(batch, "_WORKER", {"analysis_root": tmp_path / "a", "references": [], "cropper": None})

    def outcomes(result=None, error=None):
        def fake(*args, **kwargs):
            if error:
                raise error
            return result
        monkeypatch.setattr(pipeline, "analyze_clip", fake)
        return batch._analyze_task({"video_id": ids[0], "video": video})

    ok = outcomes({"segments": [[0, 100]], "proposals": [{"contact_frame": 40, "bounce_frame": 52, "zone": "ad_deep", "landing_pixel": [1, 1]}]})
    assert ok["status"] == "done" and ok["proposals"] == 1 and ok["validatable"] == 1
    assert outcomes({"segments": [], "proposals": []})["status"] == "no_court_view"
    missing = outcomes(error=ValueError("This clip has no fixed calibration; add a camera reference first."))
    assert missing["status"] == "needs_reference" and "camera reference" in missing["error"]
    crashed = outcomes(error=RuntimeError("boom"))
    assert crashed["status"] == "failed" and crashed["error"] == "RuntimeError: boom"


def test_the_status_report_counts_analyzed_cards_and_problems_per_match(tmp_path):
    project, ids = make_clip_project(tmp_path)
    analysis, matches_dir = tmp_path / "analysis", tmp_path / "matches"
    video = project["videos"][ids[0]]
    pipeline._mark_done(analysis / ids[0], pipeline.source_identity(video), [], [], 20.0)
    marker = json.loads((analysis / ids[0] / "done.json").read_text())
    marker.update({"proposals": 12, "validatable": 5, "segments": 2})
    (analysis / ids[0] / "done.json").write_text(json.dumps(marker))
    batch.write_ledger(analysis, {"clips": {ids[1]: {"status": "needs_reference"}}})
    (row,) = batch.status_report(project, matches_dir, analysis)
    assert row == {"match_id": "proj_match", "clips": 2, "analyzed": 1, "proposals": 12, "cards": 5,
                   "needs_reference": 1, "no_court_view": 0, "failed": 0}
    assert batch.read_ledger(analysis)["updated"]
    assert batch.read_ledger(tmp_path / "none") == {"clips": {}}


@pytest.mark.skipif(sys.platform != "win32", reason="process priority classes are a Windows feature")
def test_workers_really_drop_to_below_normal_priority():
    code = (
        "import ctypes; from ctypes import wintypes; from shot_predict.autopilot.batch import lower_priority; "
        "k = ctypes.WinDLL('kernel32'); k.GetCurrentProcess.restype = wintypes.HANDLE; "
        "k.GetPriorityClass.argtypes = [wintypes.HANDLE]; k.GetPriorityClass.restype = wintypes.DWORD; "
        "before = k.GetPriorityClass(k.GetCurrentProcess()); lower_priority(); "
        "print(before, k.GetPriorityClass(k.GetCurrentProcess()))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60, check=True).stdout.split()
    assert int(out[0]) == 0x20 and int(out[1]) == batch.BELOW_NORMAL_PRIORITY  # NORMAL before, BELOW_NORMAL after
