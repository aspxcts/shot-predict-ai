"""Import every clip in a folder into the project (idempotent; never changes existing videos or labels)."""

import argparse
import csv
from pathlib import Path

from ..storage import add_video, backup_project, read_project, video_id_for, write_project

VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv")


def clip_times(folder: Path) -> dict[str, tuple[float, float]]:
    """Optional clip_times.csv rows: name,start_seconds,end_seconds in the original recording."""
    path = folder / "clip_times.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as handle:
        return {row[0]: (float(row[1]), float(row[2])) for row in csv.reader(handle) if len(row) >= 3}


def import_folder(
    project_path: Path, folder: Path, match_id: str, near_player: str = "", far_player: str = "",
    dry_run: bool = False,
) -> dict:
    folder = folder.expanduser().resolve(strict=True)
    clips = sorted(p for p in folder.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    if not clips:
        raise ValueError(f"No video clips found in {folder}.")
    project = read_project(project_path)
    shots_before = len(project["shots"])
    added, skipped = [], []
    for clip in clips:
        if video_id_for(clip) in project["videos"]:
            skipped.append(clip.name)
        elif dry_run:
            added.append(clip.name)
        else:
            add_video(project, clip, match_id, near_player, far_player)
            added.append(clip.name)
    backup = None
    if added and not dry_run:
        # Back up first, then write; labels are untouched either way.
        backup = backup_project(project_path)
        if len(project["shots"]) != shots_before:
            raise ValueError("Importing videos must not change labels.")
        write_project(project_path, project)
    return {
        "added": added, "already_imported": skipped, "backup": str(backup) if backup else None,
        "clip_times": clip_times(folder), "dry_run": dry_run,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--match-id", required=True, help="Shared ID for every clip of one match.")
    parser.add_argument("--project", type=Path, default=Path("data") / "project.json")
    parser.add_argument("--near-player", default="")
    parser.add_argument("--far-player", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        result = import_folder(
            args.project, args.folder, args.match_id, args.near_player, args.far_player, args.dry_run,
        )
    except (ValueError, OSError) as exc:
        parser.exit(1, f"Import failed: {exc}\n")
    verb = "Would add" if result["dry_run"] else "Added"
    print(f"{verb} {len(result['added'])} clips; {len(result['already_imported'])} were already imported.")
    if result["backup"]:
        print(f"Backed up the previous project to {result['backup']}")


if __name__ == "__main__":
    main()
