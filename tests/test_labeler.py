import copy
from pathlib import Path

import pytest

from shot_predict.court import Court, ZONES
from shot_predict.dataset import ShotDataset
from shot_predict.labeler import apply_action, court_grid
from shot_predict.model import ModelConfig
from shot_predict.storage import add_video


@pytest.mark.parametrize(("hitter", "zone", "shot_type"), [
    ("near", "ad_deep", "groundstroke"), ("far", "deuce_short", "serve"),
])
def test_save_label_and_recalibrate_without_changing_old_labels(synthetic_project, hitter, zone, shot_type):
    _, project = synthetic_project
    video = project["videos"]["training"]
    court = Court(video["corners"])
    event = {
        "video_id": "training", "action": "save_shot", "contact_frame": 31,
        "bounce_frame": 45, "hitter": hitter, "shot_type": shot_type,
        "landing_pixel": list(court.target(zone, hitter)),
    }
    result = apply_action(project, "training", event)
    assert zone.replace("_", " ") in result["message"]
    labeled = copy.deepcopy(project["shots"][-1])
    assert labeled["zone"] == zone
    assert labeled["court_corners"] == video["corners"]
    corners = [[25, 108], [135, 108], [135, 12], [25, 12]]
    apply_action(project, "training", {
        "video_id": "training", "action": "save_calibration", "points": corners,
    })
    assert video["corners"] == corners
    assert project["shots"][-1] == labeled
    data = ShotDataset(
        project, [labeled], ModelConfig(image_size=32, samples=2, seconds=0.1, hidden=8), [100]
    )
    assert data[0][1] == ZONES.index(zone)
    assert len(court_grid(corners)) == 9


def test_invalid_actions_do_not_change_labels(synthetic_project):
    _, project = synthetic_project
    video = project["videos"]["training"]
    base = {
        "video_id": "training", "action": "save_shot", "contact_frame": 31,
        "bounce_frame": 45, "hitter": "near", "shot_type": "groundstroke",
        "landing_pixel": list(Court(video["corners"]).target("ad_deep", "near")),
    }
    original = copy.deepcopy(project)
    for event in [
        {**base, "video_id": "validation"},
        {**base, "contact_frame": True},
        {**base, "contact_frame": 10},
        {**base, "bounce_frame": 20},
        {**base, "landing_pixel": [-1, 0]},
        {**base, "landing_pixel": [float("nan"), 0]},
        {**base, "hitter": "unknown"},
        {"action": "save_calibration", "video_id": "training", "points": [[0, 0]] * 4},
        {"action": "undo", "video_id": "training", "shot_id": "not-last"},
        {"action": "unknown", "video_id": "training"},
    ]:
        with pytest.raises(ValueError):
            apply_action(project, "training", event)
        assert project == original


def test_undo_only_removes_latest_label_for_selected_clip(synthetic_project):
    _, project = synthetic_project
    other_shots = [shot for shot in project["shots"] if shot["video_id"] == "validation"]
    latest = [shot for shot in project["shots"] if shot["video_id"] == "training"][-1]
    apply_action(project, "training", {
        "action": "undo", "video_id": "training", "shot_id": latest["id"],
    })
    assert all(shot["id"] != latest["id"] for shot in project["shots"])
    assert [shot for shot in project["shots"] if shot["video_id"] == "validation"] == other_shots


def test_import_video_preserves_metadata_and_rejects_duplicates(synthetic_project):
    _, existing = synthetic_project
    source = Path(existing["videos"]["training"]["path"])
    project = {"version": 1, "videos": {}, "shots": []}
    video_id = add_video(project, source, " match ", " Near ", " Far ")
    video = project["videos"][video_id]
    assert video["path"] == str(source.resolve())
    assert (video["match_id"], video["near_player"], video["far_player"]) == ("match", "Near", "Far")
    assert video["frames"] == 80 and video["corners"] is None
    with pytest.raises(ValueError, match="already imported"):
        add_video(project, source, "match")
    with pytest.raises(ValueError, match="match ID"):
        add_video(project, source, " ")
