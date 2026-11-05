import os
from pathlib import Path
from uuid import uuid4

import streamlit as st

from shot_predict import labeler
from shot_predict.autopilot import matches as match_library, review
from shot_predict.court import ZONES
from shot_predict.storage import add_video, backup_project, read_project, write_project

ROOT = Path(__file__).resolve().parent
PROJECT_PATH = Path(os.environ.get("SHOT_LAB_PROJECT", ROOT / "data" / "project.json")).expanduser().resolve()
ANALYSIS_ROOT = Path(os.environ.get("SHOT_LAB_ANALYSIS", ROOT / "data" / "analysis")).expanduser().resolve()
MATCHES_DIR = Path(os.environ.get("SHOT_LAB_MATCHES", ROOT / "data" / "matches")).expanduser().resolve()
CAMERA_LIBRARY = Path(os.environ.get("SHOT_LAB_CAMERAS", ROOT / "data" / "camera" / "references.json")).expanduser().resolve()


@st.cache_resource
def backup_once(path: str) -> str:
    # One content-addressed copy per server process, taken before any label is written.
    return str(backup_project(Path(path)) or "")


st.set_page_config(page_title="Tennis shot lab", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    [data-testid="stHeader"] { height: 2rem; }
    [data-testid="stMainBlockContainer"] { padding: 2.1rem 0.6rem 0.2rem; max-width: none; }
    [data-testid="stSidebar"] [data-testid="stMainBlockContainer"] { padding: 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

try:
    project = read_project(PROJECT_PATH)
    backup_once(str(PROJECT_PATH))
    # Matches registered with the autopilot CLI appear here by themselves. This app is the only writer of the project.
    library_stamp = match_library.library_stamp(MATCHES_DIR)
    if st.session_state.get("library_stamp") != library_stamp:
        new_clips = match_library.sync_library(project, MATCHES_DIR)
        if new_clips:
            write_project(PROJECT_PATH, project)
        st.session_state.library_stamp = library_stamp
except (ValueError, OSError) as exc:
    st.error(f"Cannot load project: {exc}")
    st.stop()

if st.session_state.get("workspace_video") not in project["videos"]:
    st.session_state.workspace_video = next(iter(project["videos"]), None)

with st.sidebar:
    st.header("Project")
    st.caption("Use permitted footage. The app stores videos and labels locally.")
    with st.expander("Import a video", expanded=not project["videos"]):
        match_id = st.text_input("Match ID", help="Use the same ID for all clips from one match.")
        near_player = st.text_input("Near player (optional)")
        far_player = st.text_input("Far player (optional)")
        local_path = st.text_input("Existing local video path")
        upload = st.file_uploader("Or upload a short clip", type=["mp4", "mov", "avi", "mkv"])
        if st.button("Import video"):
            if not match_id.strip() or (not local_path.strip() and upload is None):
                st.error("Enter a match ID and choose a video.")
            else:
                copied_path = None
                imported = False
                try:
                    if local_path.strip():
                        path = Path(local_path.strip().strip('"')).expanduser().resolve()
                    else:
                        folder = ROOT / "data" / "videos"
                        folder.mkdir(parents=True, exist_ok=True)
                        path = folder / f"{uuid4().hex}{Path(upload.name).suffix.lower()}"
                        copied_path = path
                        with path.open("wb") as handle:
                            handle.write(upload.getbuffer())
                    video_id = add_video(project, path, match_id, near_player, far_player)
                    write_project(PROJECT_PATH, project)
                    imported = True
                    st.session_state.workspace_video = video_id
                    st.rerun()
                except (ValueError, OSError) as exc:
                    if copied_path is not None and not imported:
                        copied_path.unlink(missing_ok=True)
                    st.error(str(exc))

    video_id = st.session_state.workspace_video
    st.caption(
        f"{len(project['shots'])} labels | "
        f"{len({v['match_id'] for v in project['videos'].values()})} matches"
    )
    if video_id:
        with st.expander("Review labels"):
            shots = [shot for shot in project["shots"] if shot["video_id"] == video_id]
            st.dataframe(
                [{"contact": s["contact_frame"], "bounce": s["bounce_frame"],
                  "hitter": s["hitter"], "type": s["shot_type"], "zone": s["zone"]} for s in shots],
                width="stretch",
            )
            if shots:
                remove = st.selectbox(
                    "Label to delete", [s["id"] for s in reversed(shots)],
                    format_func=lambda key: next(
                        f"Frame {s['contact_frame']} - {s['zone']}" for s in shots if s["id"] == key
                    ),
                )
                if st.button("Delete selected label"):
                    project["shots"] = [s for s in project["shots"] if s["id"] != remove]
                    try:
                        write_project(PROJECT_PATH, project)
                        st.rerun()
                    except OSError as exc:
                        st.error(str(exc))
            st.write("All-project zone counts:", {
                zone: sum(s["zone"] == zone for s in project["shots"]) for zone in ZONES
            })
        with st.expander("Training and export"):
            st.write(f"Current video ID: `{video_id}`")
            st.code(
                '.\\.venv\\Scripts\\python.exe -m shot_predict.train --pretrained --freeze-backbone',
                language="powershell",
            )
            st.code(
                f'.\\.venv\\Scripts\\python.exe -m shot_predict.predict --video-id {video_id} '
                '--hitter both --output runs\\prediction.mp4',
                language="powershell",
            )
            st.caption("See README for evaluation, permissions, and current model limitations.")

if not project["videos"]:
    st.title("Tennis shot lab")
    st.info("Open the sidebar and import a match video to begin.")
    st.stop()

video_id = st.session_state.workspace_video
video = project["videos"][video_id]
if not Path(video["path"]).is_file():
    st.error(f"Source video is missing: {video['path']}. Restore it or update its path in the project file.")
    st.stop()
try:
    url = labeler.video_url(video["path"])
    grid = labeler.court_grid(video["corners"])
except (ValueError, OSError, RuntimeError) as exc:
    st.error(str(exc))
    st.stop()

shots = [shot for shot in project["shots"] if shot["video_id"] == video_id]
review_payload = review.load_review(project, video_id, ANALYSIS_ROOT)
event = labeler.workspace(
    video_id=video_id, video_url=url, fps=video["fps"], frames=video["frames"],
    width=video["width"], height=video["height"], corners=video["corners"], grid=grid,
    videos=[
        {"id": key, "name": f"{Path(item['path']).name} - {item['match_id']}"}
        for key, item in project["videos"].items()
    ],
    shot_count=len(shots), total_shots=len(project["shots"]),
    last_shot_id=shots[-1]["id"] if shots else None,
    review=review_payload,
    acknowledgement=st.session_state.get("workspace_ack"),
)
if event is not None:
    if not isinstance(event, dict) or not isinstance(event.get("event_id"), str):
        st.error("The labeling workspace sent an invalid action. Refresh the page.")
        st.stop()
    event_id = event["event_id"]
    if event_id != st.session_state.get("workspace_last_event"):
        st.session_state.workspace_last_event = event_id
        action = event.get("action")
        try:
            if action == "select_video":
                if event.get("video_id") != video_id or event.get("selected_video") not in project["videos"]:
                    raise ValueError("That clip is no longer available.")
                st.session_state.workspace_video = event["selected_video"]
                result = {"message": "Clip selected."}
            else:
                result = labeler.apply_action(project, video_id, event, ANALYSIS_ROOT, CAMERA_LIBRARY)
                write_project(PROJECT_PATH, project)
            ack = {"event_id": event_id, "action": action, "ok": True, **result}
        except (ValueError, OSError) as exc:
            ack = {"event_id": event_id, "action": action, "ok": False, "message": str(exc)}
        st.session_state.workspace_ack = ack
        st.rerun()
