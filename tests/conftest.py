import cv2
import numpy as np
import pytest

from shot_predict.court import Court
from shot_predict.storage import add_shot, write_project
from shot_predict.video import metadata


@pytest.fixture
def synthetic_project(tmp_path):
    video_path = tmp_path / "match.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 20, (160, 120))
    assert writer.isOpened()
    for index in range(80):
        image = np.full((120, 160, 3), (index * 3) % 255, np.uint8)
        cv2.rectangle(image, (20, 10), (140, 110), (70, 170, 70), 2)
        cv2.circle(image, (20 + index % 120, 60), 3, (0, 255, 255), -1)
        writer.write(image)
    writer.release()
    court = Court([[20, 110], [140, 110], [140, 10], [20, 10]])
    project = {"version": 1, "videos": {}, "shots": []}
    for video_id, match_id in [("training", "train-match"), ("validation", "validation-match")]:
        project["videos"][video_id] = {
            "path": str(video_path), "match_id": match_id, "corners": court.corners,
            "near_player": f"{match_id}-near", "far_player": f"{match_id}-far",
            **metadata(video_path),
        }
        add_shot(project, video_id, 30, 42, "near", "groundstroke", court.target("ad_deep", "near"))
        add_shot(project, video_id, 50, 65, "far", "serve", court.target("deuce_short", "far"))
    # Reusing synthetic footage here tests split mechanics only, not model accuracy.
    project_path = tmp_path / "project.json"
    write_project(project_path, project)
    return project_path, project
