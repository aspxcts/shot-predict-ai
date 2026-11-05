import json

import cv2
import numpy as np
import pytest

from shot_predict.autopilot.importer import import_folder
from shot_predict.storage import read_project, video_id_for


def make_clip(path, frames=12):
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((48, 64, 3), index * 10, np.uint8))
    writer.release()


def test_import_is_idempotent_and_preserves_existing_videos_and_labels(tmp_path, synthetic_project):
    project_path, original = synthetic_project
    folder = tmp_path / "match"
    folder.mkdir()
    for name in ("clip_001.avi", "clip_000.avi"):
        make_clip(folder / name)
    (folder / "notes.txt").write_text("ignored")
    (folder / "clip_times.csv").write_text("clip_000.avi,0.0,10.5\nclip_001.avi,10.5,20.0\n")

    preview = import_folder(project_path, folder, "new-match", dry_run=True)
    assert preview["added"] == ["clip_000.avi", "clip_001.avi"] and preview["dry_run"]
    assert read_project(project_path)["videos"].keys() == original["videos"].keys()

    result = import_folder(project_path, folder, "new-match", far_player="Someone")
    assert result["added"] == ["clip_000.avi", "clip_001.avi"] and result["already_imported"] == []
    assert result["backup"] and result["clip_times"]["clip_001.avi"] == (10.5, 20.0)
    updated = read_project(project_path)
    assert updated["shots"] == original["shots"]
    for key, video in original["videos"].items():
        assert updated["videos"][key] == video
    new = updated["videos"][video_id_for(folder / "clip_000.avi")]
    assert new["match_id"] == "new-match" and new["corners"] is None and new["far_player"] == "Someone"
    assert new["frames"] == 12 and new["fps"] == pytest.approx(30)

    before = project_path.read_bytes()
    again = import_folder(project_path, folder, "new-match")
    assert again["added"] == [] and len(again["already_imported"]) == 2 and again["backup"] is None
    assert project_path.read_bytes() == before


def test_import_rejects_empty_folders_and_missing_match_id(tmp_path, synthetic_project):
    project_path, _ = synthetic_project
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="No video clips"):
        import_folder(project_path, empty, "m")
    clip = tmp_path / "one"
    clip.mkdir()
    make_clip(clip / "a.avi")
    with pytest.raises(ValueError, match="match ID"):
        import_folder(project_path, clip, "  ")
    assert json.loads(project_path.read_text())["videos"].keys() == {"training", "validation"}
