import cv2
import numpy as np
import pytest

from shot_predict.autopilot import camera
from shot_predict.autopilot.camera import CameraTrack, build_reference, match_reference

SIZE = (1280, 720)
CORNERS = np.array([[330, 600], [950, 600], [820, 250], [460, 250]], np.float32)


def textured_background(seed):
    rng = np.random.default_rng(seed)
    image = cv2.GaussianBlur(rng.integers(0, 255, (SIZE[1], SIZE[0], 3), dtype=np.uint8), (0, 0), 3)
    image = cv2.addWeighted(image, 0.5, np.full_like(image, (70, 130, 80)), 0.5, 0)
    for _ in range(400):
        x, y = int(rng.integers(0, SIZE[0] - 40)), int(rng.integers(0, SIZE[1] - 40))
        w, h = int(rng.integers(6, 40)), int(rng.integers(6, 40))
        cv2.rectangle(image, (x, y), (x + w, y + h), tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    for _ in range(40):
        cv2.putText(image, "ADS%d" % rng.integers(100), (int(rng.integers(0, SIZE[0] - 120)), int(rng.integers(20, SIZE[1]))),
                    cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(0.5, 1.4)), (255, 255, 255), 2)
    return image


def scene(seed=1, lines=True):
    image = textured_background(seed)
    cv2.fillConvexPoly(image, CORNERS.astype(np.int32), (170, 110, 60))
    if lines:
        for a, b in zip(camera.court_lines(CORNERS, 80)[:-1], camera.court_lines(CORNERS, 80)[1:]):
            if np.linalg.norm(a - b) < 40:
                cv2.line(image, tuple(map(int, a)), tuple(map(int, b)), (245, 245, 245), 3)
    return image


def warp(image, scale=1.0, shift=(0.0, 0.0), angle=0.0):
    centre = (SIZE[0] / 2, SIZE[1] / 2)
    rotation = cv2.getRotationMatrix2D(centre, angle, scale)
    rotation[:, 2] += shift
    homography = np.vstack([rotation, [0, 0, 1]]).astype(np.float64)
    return cv2.warpPerspective(image, homography, SIZE, borderValue=(60, 120, 70)), homography


def expected(homography):
    return cv2.perspectiveTransform(CORNERS.reshape(-1, 1, 2), homography.astype(np.float32)).reshape(4, 2)


@pytest.fixture(scope="module")
def reference():
    return build_reference("main", scene(), CORNERS)


@pytest.fixture(scope="module")
def detector():
    return camera.make_detector()


def test_pan_zoom_and_rotation_recover_the_court_corners(reference, detector):
    for scale, shift, angle in [(1.0, (0, 0), 0), (1.15, (-40, 15), 0), (0.85, (60, -20), 1.5), (1.3, (-120, -60), -2)]:
        frame, homography = warp(scene(), scale, shift, angle)
        match = match_reference(frame, reference, detector)
        assert match.accepted, (scale, shift, angle, match.reason)
        assert np.abs(match.corners - expected(homography)).max() < 3.0
        assert match.line_score > 0.8 and match.inliers >= camera.MIN_INLIERS


def test_unrelated_frames_are_rejected_as_non_court_views(reference, detector):
    assert not match_reference(textured_background(99), reference, detector).accepted
    blank = np.full((SIZE[1], SIZE[0], 3), 90, np.uint8)
    assert match_reference(blank, reference, detector).reason in ("too_few_features", "too_few_matches")


def test_same_background_without_court_lines_fails_the_line_check(reference, detector):
    frame, _ = warp(scene(lines=False), 1.1, (-20, 10), 0)
    match = match_reference(frame, reference, detector)
    assert not match.accepted and match.reason == "court_lines_not_found"


def test_graphics_overlays_are_masked_and_scale_with_resolution():
    mask = camera.graphics_mask(1920, 1080)
    assert mask[950, 100] == 0 and mask[50, 1800] == 0 and mask[540, 960] == 255 and mask[300, 300] == 255
    small = camera.graphics_mask(960, 540)
    assert small[475, 50] == 0 and small[270, 480] == 255


def test_plausibility_rejects_flipped_tiny_and_non_convex_courts():
    size = (1280, 720)
    assert camera.plausible(CORNERS.copy(), size)
    assert not camera.plausible(CORNERS[::-1].copy(), size)
    assert not camera.plausible(CORNERS[[1, 0, 3, 2]].copy(), size)
    assert not camera.plausible(np.array([[10, 12], [14, 12], [13, 10], [11, 10]], np.float32), size)
    bowtie = CORNERS.copy()
    bowtie[[2, 3]] = bowtie[[3, 2]]
    assert not camera.plausible(bowtie, size)
    assert not camera.plausible(np.full((4, 2), np.nan), size)


def sample(frame, accepted=True, x=0.0, reference="main"):
    corners = [[x, 100.0], [x + 100, 100.0], [x + 90, 20.0], [x + 10, 20.0]]
    return {"frame": frame, "accepted": accepted, "reference": reference if accepted else None,
            "corners": corners if accepted else None, "reason": "ok" if accepted else "too_few_matches",
            "inliers": 50, "line_score": 0.9}


def test_track_interpolates_inside_views_and_never_bridges_cuts():
    track = CameraTrack(
        [sample(0, x=0), sample(30, x=30), sample(60, False), sample(90, False), sample(120, x=10), sample(150, x=10)],
        fps=60.0,
    )
    assert [(s["start"], s["end"]) for s in track.segments] == [(0, 30), (120, 150)]
    assert track.in_court_view(15) and not track.in_court_view(31) and not track.in_court_view(90)
    assert track.corners_at(15)[0, 0] == pytest.approx(15.0)
    assert track.corners_at(30)[0, 0] == pytest.approx(30.0)
    assert track.corners_at(60) is None and track.corners_at(-5) is None
    assert track.corners_at(140)[0, 0] == pytest.approx(10.0)
    assert track.coverage(180)["segments"] == 2


def test_track_does_not_bridge_a_missed_sample_and_splits_on_a_different_reference():
    # A rejected sample may be a camera cut, so the view is split rather than interpolated across it.
    track = CameraTrack([sample(0), sample(30), sample(60, False), sample(90), sample(120, reference="low")], fps=60.0)
    assert [(s["start"], s["end"], s["reference"]) for s in track.segments] == [
        (0, 30, "main"), (90, 90, "main"), (120, 120, "low"),
    ]
    assert track.corners_at(75) is None and track.corners_at(90) is not None


def test_speed_is_zero_for_a_steady_camera_and_matches_the_pan_rate_otherwise():
    steady = CameraTrack([sample(0, x=5), sample(30, x=5), sample(60, x=5)], fps=60.0)
    assert steady.speed_at(15) == 0.0
    panning = CameraTrack([sample(0, x=0), sample(30, x=60), sample(60, x=60)], fps=60.0)  # 2 px per frame at first
    assert panning.speed_at(15) == pytest.approx(2.0)
    assert panning.speed_at(45) == 0.0
    assert panning.speed_at(500) is None and panning.speed_at(-40) is None


def test_track_round_trips_through_json():
    samples = [sample(0), sample(30, x=5), sample(60, False)]
    restored = CameraTrack.from_dict(CameraTrack(samples, 60.0).to_dict())
    assert restored.segments[0]["end"] == 30 and restored.corners_at(15)[0, 0] == pytest.approx(2.5)


def test_scan_clip_samples_every_half_second_and_stops_at_the_end(tmp_path, reference):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, SIZE)
    good, other = scene(), textured_background(5)
    for index in range(30):
        writer.write(good if index < 12 else other)
    writer.release()
    samples = camera.scan_clip(path, [reference], 10.0, 30)
    assert [s["frame"] for s in samples] == [0, 5, 10, 15, 20, 25]
    assert [s["accepted"] for s in samples] == [True, True, True, False, False, False]
    assert samples[0]["corners"] is not None and samples[3]["corners"] is None


def test_library_round_trip_rebuilds_a_reference_from_the_source_frame(tmp_path):
    path = tmp_path / "ref.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, SIZE)
    for _ in range(5):
        writer.write(scene())
    writer.release()
    project = {"videos": {"v": {"path": str(path)}}}
    library = tmp_path / "camera" / "references.json"
    entries = [{"name": "main", "video_id": "v", "frame": 2, "corners": CORNERS.tolist(), "status": "unconfirmed"}]
    camera.write_library_entries(library, entries)
    assert camera.read_library_entries(library) == entries
    (reference,) = camera.load_references(camera.read_library_entries(library), project)
    assert reference.name == "main" and len(reference.keypoints) > 100
    with pytest.raises(ValueError, match="unique"):
        camera.write_library_entries(library, entries * 2)
    with pytest.raises(ValueError, match="no longer in the project"):
        camera.load_references([{**entries[0], "video_id": "gone"}], project)
    assert camera.read_library_entries(tmp_path / "missing.json") == []
