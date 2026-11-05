"""Command line for the offline analysis that proposes shots. Labels are never changed.

  python -m shot_predict.autopilot.cli add-match --source "C:\\path\\full_match.mp4" --match-id usopen_2026_r4 [--near X --far Y]
  python -m shot_predict.autopilot.cli batch [--match-id ID] [--workers 3] [--limit N]
  python -m shot_predict.autopilot.cli status
  python -m shot_predict.autopilot.cli analyze --match-id paul_alcaraz_us_open_2026 [--clips 3]
  python -m shot_predict.autopilot.cli add-reference --video-id ID --frame N --corners x,y,x,y,x,y,x,y --name NAME

add-match splits a full recording into clips (or adopts a folder of clips), batch analyzes every pending clip with
several worker processes, and the labeling app picks both up by itself: open Validate mode and go.
"""

import argparse
import json
import sys
from pathlib import Path

from ..storage import read_project
from . import batch, camera, matches, pipeline


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", type=Path, default=Path("data") / "project.json")
    p.add_argument("--analysis", type=Path, default=Path("data") / "analysis")
    p.add_argument("--library", type=Path, default=Path("data") / "camera" / "references.json")
    p.add_argument("--matches", type=Path, default=Path("data") / "matches")
    p.add_argument("--videos-root", type=Path, default=Path("data") / "videos")
    sub = p.add_subparsers(dest="command", required=True)
    m = sub.add_parser("add-match", help="Register a match: split a recording into clips, or adopt a folder of clips.")
    m.add_argument("--source", type=Path, required=True, help="A video file (split into clips) or a folder of clips (used in place).")
    m.add_argument("--match-id", required=True)
    m.add_argument("--near-player", default="")
    m.add_argument("--far-player", default="")
    m.add_argument("--clip-seconds", type=int, default=matches.DEFAULT_CLIP_SECONDS)
    m.add_argument("--reencode", action="store_true", help="Re-encode to H.264 instead of copying the video stream.")
    b = sub.add_parser("batch", help="Analyze every pending clip with several worker processes (resumable).")
    b.add_argument("--match-id", default=None, help="Only this match.")
    b.add_argument("--workers", type=int, default=3, help="Parallel clips. Each needs about 1.5 GB of memory.")
    b.add_argument("--threads", type=int, default=2, help="CPU threads per worker.")
    b.add_argument("--limit", type=int, default=None, help="Only the first N pending clips.")
    a = sub.add_parser("analyze", help="Analyze clips one at a time (resumable).")
    a.add_argument("--video-id", action="append", default=[])
    a.add_argument("--match-id", default=None)
    a.add_argument("--clips", type=int, default=None, help="Only the first N matching clips.")
    a.add_argument("--seconds", nargs=2, type=float, metavar=("START", "END"), default=None,
                   help="Only this time range (requires exactly one clip).")
    a.add_argument("--threads", type=int, default=4)
    sub.add_parser("status", help="Show which clips have been analyzed.")
    r = sub.add_parser("add-reference", help="Register a camera reference from a calibrated frame.")
    r.add_argument("--video-id", required=True)
    r.add_argument("--frame", type=int, required=True)
    r.add_argument("--corners", required=True, help="Eight numbers: near-left, near-right, far-right, far-left (x,y each).")
    r.add_argument("--name", required=True)
    return p


def select_videos(project, args):
    ids = args.video_id or [
        key for key, video in project["videos"].items() if args.match_id in (None, video["match_id"])
    ]
    unknown = [i for i in ids if i not in project["videos"]]
    if unknown:
        raise ValueError(f"Unknown video IDs: {unknown}")
    ids = sorted(ids, key=lambda i: project["videos"][i]["path"])
    return ids[: args.clips] if args.clips else ids


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        project = read_project(args.project)
        if args.command == "add-match":
            result = matches.register_match(
                args.source, args.match_id, args.matches, args.videos_root, args.near_player, args.far_player,
                args.clip_seconds, args.reencode)
            how = "split into" if result["split"] else "found"
            print(f"Match {result['match_id']}: {how} {result['clips']} clips in {result['clips_folder']}.")
            print("Next: run the batch (python -m shot_predict.autopilot.cli batch), then open the labeler and press Validate.")
            return 0
        if args.command == "batch":
            summary = batch.run_batch(args.project, args.matches, args.analysis, args.library, args.workers,
                                      args.threads, args.match_id, args.limit, log=lambda m: print(m, flush=True))
            print(json.dumps(summary))
            return 0 if not summary.get("failed") else 2
        if args.command == "add-reference":
            numbers = [float(v) for v in args.corners.split(",")]
            if len(numbers) != 8:
                raise ValueError("--corners needs eight numbers.")
            corners = [numbers[i:i + 2] for i in range(0, 8, 2)]
            from ..court import Court

            Court(corners)
            if args.video_id not in project["videos"]:
                raise ValueError("Unknown video ID.")
            entries = camera.read_library_entries(args.library)
            entries.append({"name": args.name, "video_id": args.video_id, "frame": args.frame, "corners": corners,
                            "status": "confirmed_by_user"})
            camera.write_library_entries(args.library, entries)
            print(f"Registered camera reference {args.name}.")
            return 0
        if args.command == "status":
            report = batch.status_report(project, args.matches, args.analysis)
            for row in report:
                notes = [f"{row[k]} {label}" for k, label in (("needs_reference", "need a camera reference"),
                         ("no_court_view", "show no court view"), ("failed", "failed")) if row[k]]
                print(f"{row['match_id']}: {row['analyzed']}/{row['clips']} clips analyzed | {row['proposals']} proposals | "
                      f"{row['cards']} validation cards" + (f" | {', '.join(notes)}" if notes else ""))
            print(f"{sum(r['analyzed'] for r in report)} of {sum(r['clips'] for r in report)} clips analyzed "
                  f"in {len(report)} matches.")
            return 0
        import cv2
        import torch

        torch.set_num_threads(args.threads)
        cv2.setNumThreads(args.threads)
        ids = select_videos(project, args)
        if args.seconds and len(ids) != 1:
            raise ValueError("--seconds needs exactly one clip (use --video-id).")
        from ..players import PlayerCropper

        cropper = PlayerCropper()
        entries = camera.read_library_entries(args.library)
        references = camera.load_references(entries, project) if entries else []
        for number, video_id in enumerate(ids, 1):
            video = project["videos"][video_id]
            frame_range = None
            if args.seconds:
                frame_range = (round(args.seconds[0] * video["fps"]), round(args.seconds[1] * video["fps"]))
            print(f"Clip {number}/{len(ids)}: {Path(video['path']).name}", flush=True)
            pipeline.analyze_clip(project, video_id, args.analysis, references, cropper, frame_range, log=lambda m: print(m, flush=True))
        return 0
    except (ValueError, OSError) as exc:
        print(f"Analysis failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
