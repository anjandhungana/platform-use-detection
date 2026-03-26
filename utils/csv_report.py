from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path
from typing import Any


def _safe_base_name(source_video_name: str) -> str:
    base = Path(source_video_name).stem.strip()
    return base or "analysis"


def _write_timeline_csv(path: str, per_second_records: list[dict[str, Any]]) -> None:
    fieldnames = ["second_index", "timestamp_sec", "unique_bird_ids"]
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for record in per_second_records:
            writer.writerow(
                {
                    "second_index": int(record.get("second_index", 0)),
                    "timestamp_sec": float(record.get("timestamp_sec", 0.0)),
                    "unique_bird_ids": int(
                        record.get("unique_bird_ids", record.get("unique_ids_active", 0))
                    ),
                }
            )


def _write_id_stats_csv(path: str, per_id_records: list[dict[str, Any]]) -> None:
    fieldnames = [
        "track_id",
        "class_label",
        "first_frame",
        "last_frame",
        "duration_frames",
        "duration_seconds",
    ]
    with open(path, "w", newline="", encoding="utf-8") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for record in per_id_records:
            writer.writerow(
                {
                    "track_id": int(record.get("track_id", 0)),
                    "class_label": str(record.get("class_label", "")),
                    "first_frame": int(record.get("first_frame", 0)),
                    "last_frame": int(record.get("last_frame", 0)),
                    "duration_frames": int(record.get("duration_frames", 0)),
                    "duration_seconds": round(
                        float(record.get("duration_seconds", 0.0)), 2
                    ),
                }
            )


def generate_csv_reports(
    inference_summary: dict[str, Any],
    source_video_name: str,
) -> dict[str, str]:
    """Generate timeline and ID-stat CSV reports and return their paths."""
    base_name = _safe_base_name(source_video_name)

    timeline_tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f"_{base_name}_timeline.csv",
    )
    timeline_path = timeline_tmp.name
    timeline_tmp.close()

    id_stats_tmp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=f"_{base_name}_id_stats.csv",
    )
    id_stats_path = id_stats_tmp.name
    id_stats_tmp.close()

    per_second_records = inference_summary.get("csv_per_second_records", []) or []
    per_id_records = inference_summary.get("csv_per_id_records", []) or []

    _write_timeline_csv(timeline_path, per_second_records)
    _write_id_stats_csv(id_stats_path, per_id_records)

    return {
        "timeline_path": timeline_path,
        "id_stats_path": id_stats_path,
        "timeline_filename": f"{base_name}_timeline.csv",
        "id_stats_filename": f"{base_name}_id_stats.csv",
    }
