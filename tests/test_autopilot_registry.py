import cv2
import numpy as np
import pytest

from shot_predict import labeler
from shot_predict.autopilot import camera, registry

SIZE = (1280, 720)
CORNERS = np.array([[330, 600], [950, 600], [820, 250], [460, 250]], np.float32)


def scene(lines=True):
    rng = np.random.default_rng(1)
    image = cv2.GaussianBlur(rng.integers(0, 255, (SIZE[1], SIZE[0], 3), dtype=np.uint8), (0, 0), 3)
    image = cv2.addWeighted(image, 0.5, np.full_like(image, (70, 130, 80)), 0.5, 0)
    for _ in range(400):
        x, y = int(rng.integers(0, SIZE[0] - 40)), int(rng.integers(0, SIZE[1] - 40))
        cv2.rectangle(image, (x, y), (x + int(rng.integers(6, 40)), y + int(rng.integers(6, 40))),
                      tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    for _ in range(40):
        cv2.putText(image, "ADS%d" % rng.integers(100), (int(rng.integers(0, SIZE[0] - 120)), int(rng.integers(20, SIZE[1]))),
                    cv2.FONT_HERSHEY_SIMPLEX, float(rng.uniform(0.5, 1.4)), (255, 255, 255), 2)
    cv2.fillConvexPoly(image, CORNERS.astype(np.int32), (170, 110, 60))
    if lines:
        points = camera.court_lines(CORNERS, 80)
        for a, b in zip(points[:-1], points[1:]):
            if np.linalg.norm(a - b) < 40:
                cv2.line(image, tuple(map(int, a)), tuple(map(int, b)), (245, 245, 245), 3)
    return image


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    path = tmp_path_factory.mktemp("cam") / "wide.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, SIZE)
    for _ in range(6):
        writer.write(scene())
    writer.release()
    return {"path": str(path), "frames": 6, "fps": 10.0, "width": SIZE[0], "height": SIZE[1], "match_id": "venue_match"}


def test_exact_corners_register_a_camera_that_can_be_rebuilt_from_its_path(clip, tmp_path):
    library = tmp_path / "camera" / "references.json"
    result = registry.register_reference(clip, 2, CORNERS.tolist(), library)
    assert result["name"] == "venue_match-main" and result["line_score"] >= camera.MIN_LINE_SCORE
    (entry,) = camera.read_library_entries(library)
    assert entry["path"] == clip["path"] and entry["frame"] == 2 and entry["status"] == "confirmed_by_user"
    (reference,) = camera.load_references([entry], {"videos": {}})  # no project needed: the entry carries the path
    match = camera.match_reference(scene(), reference)
    assert match.accepted and np.abs(match.corners - CORNERS).max() < 2
    second = registry.register_reference(clip, 3, CORNERS.tolist(), library)
    assert second["name"] == "venue_match-main-2"


def test_corners_that_miss_the_court_lines_are_refused_before_anything_is_saved(clip, tmp_path):
    library = tmp_path / "references.json"
    off = CORNERS + np.array([[60, 40], [-50, 40], [40, -30], [-60, -30]], np.float32)
    with pytest.raises(ValueError, match="do not line up"):
        registry.register_reference(clip, 1, off.tolist(), library)
    with pytest.raises(ValueError, match="do not look like a court"):
        registry.register_reference(clip, 1, CORNERS[::-1].tolist(), library)
    with pytest.raises(ValueError, match="outside the clip"):
        registry.register_reference(clip, 99, CORNERS.tolist(), library)
    assert not library.exists()


def test_the_labeler_action_registers_from_four_clicks_and_validates_the_input(clip, tmp_path):
    library = tmp_path / "references.json"
    project = {"version": 1, "videos": {"v": clip}, "shots": []}
    event = {"video_id": "v", "action": "save_camera_reference", "points": CORNERS.tolist(), "frame": 3}
    result = labeler.apply_action(project, "v", event, tmp_path / "analysis", library)
    assert "venue_match-main" in result["message"] and len(camera.read_library_entries(library)) == 1
    for bad in ({**event, "points": CORNERS.tolist()[:3]}, {**event, "frame": "3"}, {**event, "points": [[5, 5]] * 4},
                {**event, "points": [[-5, 5]] + CORNERS.tolist()[1:]}):
        with pytest.raises(ValueError):
            labeler.apply_action(project, "v", bad, tmp_path / "analysis", library)
    assert len(camera.read_library_entries(library)) == 1
    assert project["videos"]["v"].get("corners") is None  # registering a camera never touches the clip's own calibration
