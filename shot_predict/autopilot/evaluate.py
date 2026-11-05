"""Score proposed shots against human labels.

Nothing here reads or writes labels; it only compares two lists. Precision is a lower bound:
a real hit the human chose not to label counts as a false proposal, so by default only
proposals inside the labeled span of a clip (plus padding) are scored.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

CONTACT_TOLERANCES_MS = (50, 100, 200, 500)
GATE_TOLERANCE_MS = 100
PAIRING_WINDOW_MS = 250
BOUNCE_TOLERANCE_MS = 120
SPAN_PADDING_SECONDS = 1.0


def pair_events(true_frames: list[int], proposed_frames: list[int], tolerance: int) -> list[tuple[int, int]]:
    """One-to-one nearest pairing; returns (true_index, proposed_index) pairs within tolerance frames."""
    candidates = sorted(
        (abs(t - p), ti, pi)
        for ti, t in enumerate(true_frames) for pi, p in enumerate(proposed_frames)
        if abs(t - p) <= tolerance
    )
    used_true, used_proposed, pairs = set(), set(), []
    for _, ti, pi in candidates:
        if ti not in used_true and pi not in used_proposed:
            used_true.add(ti)
            used_proposed.add(pi)
            pairs.append((ti, pi))
    return sorted(pairs)


def frames_for(ms: float, fps: float) -> int:
    return int(ms * fps / 1000)


def evaluate_video(shots: list[dict], proposals: list[dict], fps: float, restrict_to_span: bool = True) -> dict:
    """Compare one clip's labeled shots with its proposals (both sorted by contact frame)."""
    shots = sorted(shots, key=lambda s: s["contact_frame"])
    proposals = sorted(proposals, key=lambda p: p["contact_frame"])
    if restrict_to_span and shots:
        low = shots[0]["contact_frame"] - SPAN_PADDING_SECONDS * fps
        high = shots[-1]["bounce_frame"] + SPAN_PADDING_SECONDS * fps
        scored = [p for p in proposals if low <= p["contact_frame"] <= high]
    else:
        scored = proposals
    true_frames = [s["contact_frame"] for s in shots]
    proposed_frames = [p["contact_frame"] for p in scored]
    recall = {
        ms: len(pair_events(true_frames, proposed_frames, frames_for(ms, fps)))
        for ms in CONTACT_TOLERANCES_MS
    }
    pairs = pair_events(true_frames, proposed_frames, frames_for(PAIRING_WINDOW_MS, fps))
    rows = []
    paired_true = {ti for ti, _ in pairs}
    paired_proposed = {pi for _, pi in pairs}
    for ti, pi in pairs:
        shot, proposal = shots[ti], scored[pi]
        bounce = proposal.get("bounce_frame")
        rows.append({
            "contact_frame": shot["contact_frame"], "status": "matched",
            "contact_error_ms": (proposal["contact_frame"] - shot["contact_frame"]) * 1000 / fps,
            "hitter_correct": proposal["hitter"] == shot["hitter"],
            "type_correct": proposal.get("shot_type") == shot["shot_type"],
            "bounce_error_ms": None if bounce is None else (bounce - shot["bounce_frame"]) * 1000 / fps,
            "zone_correct": proposal.get("zone") == shot["zone"],
            "true_zone": shot["zone"], "proposed_zone": proposal.get("zone"),
            "confidence": proposal.get("confidence"),
        })
    for ti, shot in enumerate(shots):
        if ti not in paired_true:
            rows.append({"contact_frame": shot["contact_frame"], "status": "missed", "true_zone": shot["zone"]})
    for pi, proposal in enumerate(scored):
        if pi not in paired_proposed:
            rows.append({
                "contact_frame": proposal["contact_frame"], "status": "unmatched_proposal",
                "proposed_zone": proposal.get("zone"), "confidence": proposal.get("confidence"),
            })
    return {
        "labeled": len(shots), "proposals_scored": len(scored), "proposals_total": len(proposals),
        "recall_matches": {str(ms): n for ms, n in recall.items()},
        "paired": len(pairs), "unmatched_proposals": len(scored) - len(pairs),
        "rows": sorted(rows, key=lambda r: r["contact_frame"]),
    }


def summarize(results: dict[str, dict]) -> dict:
    """Pool per-clip results (clip name -> evaluate_video output) into one report."""
    labeled = sum(r["labeled"] for r in results.values())
    scored = sum(r["proposals_scored"] for r in results.values())
    paired = sum(r["paired"] for r in results.values())
    unmatched = sum(r["unmatched_proposals"] for r in results.values())
    matched_rows = [row for r in results.values() for row in r["rows"] if row["status"] == "matched"]
    errors = [abs(row["contact_error_ms"]) for row in matched_rows]
    bounce_errors = [abs(row["bounce_error_ms"]) for row in matched_rows if row["bounce_error_ms"] is not None]
    summary = {
        "labeled_shots": labeled, "proposals_scored": scored,
        "contact_recall": {
            f"within_{ms}ms": {
                "matched": sum(r["recall_matches"][str(ms)] for r in results.values()),
                "of": labeled,
                "rate": sum(r["recall_matches"][str(ms)] for r in results.values()) / labeled if labeled else None,
            }
            for ms in CONTACT_TOLERANCES_MS
        },
        "paired_within_window": paired,
        "median_abs_contact_error_ms": statistics.median(errors) if errors else None,
        "hitter_accuracy": _rate(sum(r["hitter_correct"] for r in matched_rows), len(matched_rows)),
        "shot_type_accuracy": _rate(sum(r["type_correct"] for r in matched_rows), len(matched_rows)),
        "bounce_within_tolerance": _rate(
            sum(e <= BOUNCE_TOLERANCE_MS for e in bounce_errors), len(matched_rows)
        ),
        "median_abs_bounce_error_ms": statistics.median(bounce_errors) if bounce_errors else None,
        "zone_agreement_of_paired": _rate(sum(r["zone_correct"] for r in matched_rows), len(matched_rows)),
        "unmatched_proposals": unmatched,
        "unmatched_per_paired_shot": unmatched / paired if paired else None,
        "gates": {},
        "notes": [
            "Precision is a lower bound: unlabeled real hits count as unmatched proposals.",
            "Hitter, bounce and zone are scored on shots paired within the "
            f"{PAIRING_WINDOW_MS} ms window; missing bounces/zones count as wrong.",
        ],
    }
    recall = summary["contact_recall"][f"within_{GATE_TOLERANCE_MS}ms"]["rate"]
    summary["gates"] = {
        "contact_recall_100ms_at_least_0.80": None if recall is None else recall >= 0.80,
        "hitter_accuracy_at_least_0.95": _passes(summary["hitter_accuracy"], 0.95),
        "zone_agreement_at_least_0.70": _passes(summary["zone_agreement_of_paired"], 0.70),
        "unmatched_per_paired_at_most_0.34": (
            None if summary["unmatched_per_paired_shot"] is None else summary["unmatched_per_paired_shot"] <= 0.34
        ),
    }
    return summary


def _rate(count: int, total: int) -> dict:
    return {"count": count, "of": total, "rate": count / total if total else None}


def _passes(block: dict, threshold: float):
    return None if block["rate"] is None else block["rate"] >= threshold


def evaluate_project(
    project: dict, proposals_by_video: dict[str, list[dict]], restrict_to_span: bool = True,
    ranges: dict[str, tuple[int, int]] | None = None,
) -> dict:
    """`ranges` limits each clip to the frames that were actually analyzed (labels outside are not scored)."""
    results = {}
    for video_id, proposals in proposals_by_video.items():
        video = project["videos"][video_id]
        shots = [s for s in project["shots"] if s["video_id"] == video_id]
        if ranges and video_id in ranges:
            low, high = ranges[video_id]
            shots = [s for s in shots if low <= s["contact_frame"] < high]
        if shots:
            results[Path(video["path"]).name] = evaluate_video(shots, proposals, video["fps"], restrict_to_span)
    return {"per_clip": {k: {kk: vv for kk, vv in v.items() if kk != "rows"} for k, v in results.items()},
            "summary": summarize(results), "rows": {k: v["rows"] for k, v in results.items()}}


def load_proposals(analysis_root: Path, video_ids: list[str], with_ranges: bool = False):
    found, ranges = {}, {}
    for video_id in video_ids:
        path = analysis_root / video_id / "proposals.json"
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8"))
            found[video_id] = data["proposals"]
            if data.get("range"):
                ranges[video_id] = tuple(data["range"])
    return (found, ranges) if with_ranges else found


def main():
    parser = argparse.ArgumentParser(description="Score saved proposals against human labels.")
    parser.add_argument("--project", type=Path, default=Path("data") / "project.json")
    parser.add_argument("--analysis", type=Path, default=Path("data") / "analysis")
    parser.add_argument("--match-id", default=None, help="Only score clips from this match.")
    parser.add_argument("--output", type=Path, required=True, help="New report folder.")
    parser.add_argument("--all-proposals", action="store_true", help="Also score proposals outside the labeled span.")
    args = parser.parse_args()
    from ..storage import read_project

    if args.output.exists():
        parser.exit(1, f"{args.output} already exists. Choose a new report folder.\n")
    project = read_project(args.project)
    ids = [k for k, v in project["videos"].items() if args.match_id in (None, v["match_id"])]
    proposals, ranges = load_proposals(args.analysis, ids, with_ranges=True)
    if not proposals:
        parser.exit(1, "No proposals found; run the analysis first.\n")
    report = evaluate_project(project, proposals, restrict_to_span=not args.all_proposals, ranges=ranges)
    args.output.mkdir(parents=True)
    (args.output / "evaluation.json").write_text(
        json.dumps({k: report[k] for k in ("summary", "per_clip")}, indent=2), encoding="utf-8")
    fields = ["clip", "contact_frame", "status", "contact_error_ms", "hitter_correct", "type_correct",
              "bounce_error_ms", "true_zone", "proposed_zone", "zone_correct", "confidence"]
    with (args.output / "per_shot.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for clip, rows in report["rows"].items():
            writer.writerows({"clip": clip, **row} for row in rows)
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
