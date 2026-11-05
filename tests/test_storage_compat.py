import json

import numpy as np
import pytest

from shot_predict.court import Court, ZONES, zone_for
from shot_predict.dataset import ShotDataset
from shot_predict.model import ModelConfig
from shot_predict.players import input_spec, spec_key
from shot_predict.storage import (
    add_review, add_shot, backup_project, input_corners, read_project, write_project,
)
from shot_predict.video import metadata

CORNERS = [[20, 110], [140, 110], [140, 10], [20, 10]]
SHIFTED = [[10, 112], [150, 112], [132, 12], [28, 12]]


def labeled(project, **kwargs):
    court = Court(CORNERS)
    return add_shot(project, "training", 30, 42, "near", "groundstroke", court.target("ad_deep", "near"), **kwargs)


def test_zone_for_matches_court_zone_for_every_zone_and_end():
    court = Court(CORNERS)
    for hitter in ("near", "far"):
        for zone in ZONES:
            x, y = court.normalized(*court.target(zone, hitter))
            assert zone_for(x, y, hitter) == court.zone(x, y, hitter) == zone
    with pytest.raises(ValueError, match="opposite side"):
        zone_for(0.5, 0.2, "far")
    with pytest.raises(ValueError, match="opposite side"):
        zone_for(0.5, 0.9, "near")
    with pytest.raises(ValueError, match="outside"):
        zone_for(1.2, 0.2, "near")


def test_manual_shot_keeps_the_original_record_shape(synthetic_project):
    _, project = synthetic_project
    project["shots"].clear()
    shot = labeled(project)
    assert set(shot) == {
        "id", "video_id", "contact_frame", "bounce_frame", "hitter", "shot_type",
        "landing", "zone", "landing_pixel", "court_corners",
    }


def test_automatic_shot_requires_a_proposal_and_records_provenance(synthetic_project):
    _, project = synthetic_project
    project["shots"].clear()
    with pytest.raises(ValueError, match="proposal"):
        labeled(project, source="auto_confirmed")
    with pytest.raises(ValueError, match="source"):
        labeled(project, source="robot", proposal={})
    shot = labeled(project, source="auto_corrected", proposal={"id": "p1", "zone": "ad_short"}, review_seconds=4.2)
    assert shot["source"] == "auto_corrected" and shot["proposal"]["id"] == "p1"
    assert shot["review_seconds"] == 4.2 and shot["saved_at"].endswith("+00:00")
    assert "input_court_corners" not in shot


def test_moving_camera_uses_bounce_corners_for_landing_and_contact_corners_for_input(synthetic_project):
    _, project = synthetic_project
    project["shots"].clear()
    video = project["videos"]["training"]
    video["corners"] = None
    pixel = Court(SHIFTED).target("deuce_deep", "near")
    shot = add_shot(
        project, "training", 30, 42, "near", "groundstroke", pixel,
        corners=SHIFTED, input_corners_at_contact=CORNERS,
    )
    assert shot["zone"] == "deuce_deep"
    assert shot["court_corners"] == SHIFTED and shot["input_court_corners"] == CORNERS
    assert input_corners(shot, video) == CORNERS
    with pytest.raises(ValueError, match="Calibrate"):
        add_shot(project, "training", 50, 60, "near", "groundstroke", pixel)
    with pytest.raises(ValueError):
        add_shot(project, "training", 50, 60, "near", "groundstroke", pixel, corners=SHIFTED,
                 input_corners_at_contact=[[0, 0], [1, 1], [0, 0], [1, 1]])


def test_legacy_shots_fall_back_to_the_clip_calibration(synthetic_project):
    _, project = synthetic_project
    video = project["videos"]["training"]
    legacy = project["shots"][0]
    assert input_corners(legacy, video) == video["corners"]
    video["corners"] = None
    with pytest.raises(ValueError, match="no court calibration"):
        input_corners(legacy, video)


def test_cache_keys_are_unchanged_for_legacy_shots_and_follow_shot_corners(synthetic_project):
    _, project = synthetic_project
    config = ModelConfig(64, 2, .1, 8, True)
    indexing = ShotDataset(project, project["shots"][:1], config, [100])
    example = indexing.examples[0]
    video = project["videos"][example["shot"]["video_id"]]
    legacy = input_spec(example, video, ModelConfig(64, 2, .1, 8, True, "player_crop", "0" * 64))
    assert legacy["corners"] == video["corners"]
    moved = {**example, "shot": {**example["shot"], "input_court_corners": SHIFTED}}
    changed = input_spec(moved, video, ModelConfig(64, 2, .1, 8, True, "player_crop", "0" * 64))
    assert changed["corners"] == SHIFTED
    assert spec_key(changed) != spec_key(legacy)


def test_dataset_validates_labels_for_clips_without_a_fixed_calibration(synthetic_project):
    _, project = synthetic_project
    project["videos"]["training"]["corners"] = None
    shots = [s for s in project["shots"] if s["video_id"] == "training"]
    config = ModelConfig(64, 2, .1, 8, True)
    assert len(ShotDataset(project, shots, config, [100]).examples) == len(shots)


def test_reviews_are_replaced_per_proposal_and_validated(synthetic_project):
    _, project = synthetic_project
    add_review(project, "training", "p-1", "skipped")
    add_review(project, "training", "p-1", "rejected")
    assert [r["decision"] for r in project["reviews"]] == ["rejected"]
    with pytest.raises(ValueError):
        add_review(project, "training", "p-2", "maybe")
    with pytest.raises(ValueError):
        add_review(project, "nope", "p-2", "out")


def test_project_round_trip_adds_empty_reviews_without_changing_shots(tmp_path, synthetic_project):
    path, project = synthetic_project
    raw = json.loads(path.read_text(encoding="utf-8"))
    raw.pop("reviews", None)
    path.write_text(json.dumps(raw), encoding="utf-8")
    loaded = read_project(path)
    assert loaded["reviews"] == [] and loaded["shots"] == raw["shots"]
    assert read_project(tmp_path / "missing.json")["reviews"] == []


def test_backup_is_idempotent_for_identical_content_and_keeps_only_recent_copies(tmp_path):
    path = tmp_path / "project.json"
    folder = tmp_path / "backups"
    assert backup_project(path, folder) is None
    project = {"version": 1, "videos": {}, "shots": [], "reviews": []}
    write_project(path, project)
    first = backup_project(path, folder)
    assert first is not None and first.read_bytes() == path.read_bytes()
    assert backup_project(path, folder) is None
    for number in range(4):
        project["shots"] = [number] * (number + 1)
        write_project(path, project)
        assert backup_project(path, folder, keep=3) is not None
    assert len(list(folder.glob("project-*.json"))) == 3


def test_label_source_filter_selects_human_only_or_assisted_labels(synthetic_project):
    from shot_predict.train import LABEL_SOURCES, filter_label_sources, parser

    _, project = synthetic_project
    assert set(LABEL_SOURCES) == {"manual", "auto_confirmed", "auto_corrected"}
    assert parser().get_default("label_sources") == list(LABEL_SOURCES)
    project["shots"][0]["source"] = "auto_corrected"
    manual_only = filter_label_sources(project, ["manual"])
    assert len(manual_only["shots"]) == len(project["shots"]) - 1 and manual_only["videos"] is project["videos"]
    assert [s["source"] for s in filter_label_sources(project, ["auto_corrected"])["shots"]] == ["auto_corrected"]
    assert len(filter_label_sources(project, list(LABEL_SOURCES))["shots"]) == len(project["shots"])
    assert len(project["shots"]) == 4  # the caller's project is not modified
    with pytest.raises(ValueError, match="No labels come from"):
        filter_label_sources(project, ["auto_confirmed"])
