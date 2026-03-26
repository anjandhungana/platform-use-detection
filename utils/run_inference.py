from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO

from utils.heatmap import build_spatial_heatmap_overlay, detection_centers_from_xyxy


def _ensure_video_path(video_input: Any) -> tuple[str, bool]:
    if isinstance(video_input, (str, Path)):
        video_path = str(video_input)
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        return video_path, False

    if hasattr(video_input, "getvalue"):
        data = video_input.getvalue()
    elif hasattr(video_input, "read"):
        data = video_input.read()
    elif isinstance(video_input, (bytes, bytearray)):
        data = bytes(video_input)
    else:
        raise TypeError(
            "video_input must be a path, bytes payload, or file-like object."
        )

    if not data:
        raise ValueError("Uploaded video is empty.")

    suffix = ".mp4"
    if hasattr(video_input, "name") and isinstance(video_input.name, str):
        candidate_suffix = Path(video_input.name).suffix
        if candidate_suffix:
            suffix = candidate_suffix

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        tmp_file.write(data)
        return tmp_file.name, True


def _build_boundary_mask(
    frame_height: int,
    frame_width: int,
    boundary_points: list[dict[str, int]],
) -> np.ndarray:
    mask = np.zeros((frame_height, frame_width), dtype=np.uint8)
    polygon = np.array(
        [[int(point["x"]), int(point["y"])] for point in boundary_points],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [polygon], 255)
    return mask


def _filter_detections_by_overlap_threshold(
    detections: sv.Detections,
    boundary_mask: np.ndarray,
    overlap_threshold_pct: float,
) -> tuple[sv.Detections, int]:
    if len(detections) == 0:
        return detections, 0

    frame_height, frame_width = boundary_mask.shape[:2]
    keep_mask: list[bool] = []

    for x1, y1, x2, y2 in detections.xyxy:
        box_width = max(0.0, float(x2) - float(x1))
        box_height = max(0.0, float(y2) - float(y1))
        box_area = box_width * box_height
        if box_area <= 0:
            keep_mask.append(False)
            continue

        x1_i = int(max(0, min(frame_width, np.floor(x1))))
        y1_i = int(max(0, min(frame_height, np.floor(y1))))
        x2_i = int(max(0, min(frame_width, np.ceil(x2))))
        y2_i = int(max(0, min(frame_height, np.ceil(y2))))

        if x2_i <= x1_i or y2_i <= y1_i:
            keep_mask.append(False)
            continue

        overlap_pixels = float(cv2.countNonZero(boundary_mask[y1_i:y2_i, x1_i:x2_i]))
        overlap_pct = (overlap_pixels / box_area) * 100.0
        keep_mask.append(overlap_pct >= overlap_threshold_pct)

    keep_np = np.array(keep_mask, dtype=bool)
    filtered = detections[keep_np]
    filtered_out_count = int(len(detections) - len(filtered))
    return filtered, filtered_out_count


def _draw_boundary_overlay(
    frame: np.ndarray,
    boundary_points: list[dict[str, int]],
    color_bgr: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    if len(boundary_points) < 3:
        return frame

    polygon = np.array(
        [[int(point["x"]), int(point["y"])] for point in boundary_points],
        dtype=np.int32,
    )
    cv2.polylines(frame, [polygon], isClosed=True, color=color_bgr, thickness=thickness)
    return frame


def run_inference(
    video_input: Any,
    model_filepath: str,
    conf_threshold: float = 0.25,
    show_labels: bool = True,
    show_track_ids: bool = True,
    label_renames: dict[str, str] | None = None,
    save_output_video: bool = False,
    output_video_path: str | None = None,
    device: str | None = None,
    boundary_points: list[dict[str, int]] | None = None,
    use_boundary_filter: bool = False,
    overlap_threshold_pct: float = 0.0,
) -> dict[str, Any]:
    if not os.path.isfile(model_filepath):
        raise FileNotFoundError(f"Model file not found: {model_filepath}")

    conf_threshold = max(0.0, min(1.0, float(conf_threshold)))
    overlap_threshold_pct = max(0.0, min(100.0, float(overlap_threshold_pct)))

    video_path, cleanup_temp_video = _ensure_video_path(video_input)
    model = YOLO(model_filepath)
    tracker_config = "botsort.yaml"
    box_annotator = sv.BoxAnnotator(thickness=2)
    label_annotator = sv.LabelAnnotator()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if cleanup_temp_video:
            os.remove(video_path)
        raise ValueError(f"Unable to open video source: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    frame_index = 0
    total_detections = 0
    per_frame_counts: list[int] = []
    detection_center_points: list[tuple[int, int]] = []
    class_counts: dict[str, int] = {}
    unique_track_ids: set[int] = set()
    per_class_unique_track_ids: dict[str, set[int]] = {}
    per_second_detection_counts: dict[int, int] = {}
    per_second_active_ids: dict[int, set[int]] = {}
    per_id_metrics: dict[int, dict[str, Any]] = {}
    max_concurrent_tracks = 0
    sample_frame_bgr = None
    sample_frame_index = None
    fallback_frame_bgr = None
    fallback_frame_index = None
    sample_frame_raw_bgr = None
    fallback_frame_raw_bgr = None
    boundary_mask = None
    boundary_filter_applied = False
    boundary_filter_warning = None
    total_filtered_out = 0
    video_writer = None
    resolved_output_video_path = output_video_path
    reassigned_id_by_raw_id: dict[int, int] = {}
    next_reassigned_id = 1

    valid_boundary_points = bool(boundary_points) and len(boundary_points or []) >= 3
    if use_boundary_filter and not valid_boundary_points:
        boundary_filter_warning = (
            "Boundary filtering requested, but at least 3 points are required. "
            "Running without boundary filtering."
        )

    try:
        while True:
            success, frame = cap.read()
            if not success:
                break

            # 1) Detect/track on the original frame.
            inference_frame = frame
            if use_boundary_filter and valid_boundary_points:
                if boundary_mask is None:
                    boundary_mask = _build_boundary_mask(
                        frame_height=frame.shape[0],
                        frame_width=frame.shape[1],
                        boundary_points=boundary_points or [],
                    )

            try:
                results = model.track(
                    source=inference_frame,
                    conf=conf_threshold,
                    tracker=tracker_config,
                    persist=True,
                    verbose=False,
                    device=device,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"BoT-SORT tracking failed to initialize or run: {exc}"
                ) from exc
            result = results[0]
            detections = sv.Detections.from_ultralytics(result)

            # 2) Filter detections by boundary overlap threshold.
            if use_boundary_filter and valid_boundary_points:
                detections, filtered_out_count = _filter_detections_by_overlap_threshold(
                    detections=detections,
                    boundary_mask=boundary_mask,
                    overlap_threshold_pct=overlap_threshold_pct,
                )
                total_filtered_out += filtered_out_count
                boundary_filter_applied = True

            frame_detection_count = len(detections)
            per_frame_counts.append(frame_detection_count)
            total_detections += frame_detection_count
            detection_centers = detection_centers_from_xyxy(detections.xyxy)
            second_index = int(frame_index / fps) if fps > 0 else 0
            per_second_detection_counts[second_index] = (
                per_second_detection_counts.get(second_index, 0) + frame_detection_count
            )

            labels: list[str] = []
            tracker_ids = getattr(detections, "tracker_id", None)
            reassigned_tracker_ids: list[int | None] = []
            if tracker_ids is not None:
                tracker_id_values = (
                    tracker_ids.tolist() if hasattr(tracker_ids, "tolist") else list(tracker_ids)
                )
                for raw_track_id in tracker_id_values:
                    reassigned_id = None
                    if raw_track_id is not None:
                        try:
                            raw_id_int = int(raw_track_id)
                            if raw_id_int not in reassigned_id_by_raw_id:
                                reassigned_id_by_raw_id[raw_id_int] = next_reassigned_id
                                next_reassigned_id += 1
                            reassigned_id = reassigned_id_by_raw_id[raw_id_int]
                        except (TypeError, ValueError):
                            reassigned_id = None
                    reassigned_tracker_ids.append(reassigned_id)

            for idx, center in enumerate(detection_centers):
                cx, cy = center
                track_id_value = None
                if idx < len(reassigned_tracker_ids):
                    track_id_value = reassigned_tracker_ids[idx]

                detection_center_points.append((cx, cy))

            frame_track_ids: set[int] = set()
            if detections.class_id is not None:
                for idx, cls_id in enumerate(detections.class_id.tolist()):
                    cls_id_int = int(cls_id)
                    cls_name = model.names.get(cls_id_int, str(cls_id_int))
                    display_name = cls_name
                    if label_renames is not None:
                        mapped_name = label_renames.get(cls_name, "")
                        if mapped_name.strip():
                            display_name = mapped_name.strip()

                    # 3) Reassign IDs after filtering and use reassigned values everywhere downstream.
                    track_id_value = None
                    if idx < len(reassigned_tracker_ids):
                        track_id_value = reassigned_tracker_ids[idx]

                    if track_id_value is not None:
                        frame_track_ids.add(track_id_value)
                        unique_track_ids.add(track_id_value)
                        per_second_active_ids.setdefault(second_index, set()).add(
                            track_id_value
                        )
                        per_class_unique_track_ids.setdefault(display_name, set()).add(
                            track_id_value
                        )
                        id_record = per_id_metrics.get(track_id_value)
                        if id_record is None:
                            per_id_metrics[track_id_value] = {
                                "track_id": track_id_value,
                                "class_label": display_name,
                                "first_frame": frame_index,
                                "last_frame": frame_index,
                                "total_detections": 1,
                            }
                        else:
                            id_record["last_frame"] = frame_index
                            id_record["total_detections"] = int(
                                id_record["total_detections"]
                            ) + 1

                    label_parts: list[str] = []
                    if show_labels:
                        label_parts.append(display_name)

                    if show_track_ids and track_id_value is not None:
                        label_parts.append(f"ID:{track_id_value}")

                    if show_labels and detections.confidence is not None:
                        conf = float(detections.confidence[idx])
                        label_parts.append(f"{conf:.2f}")

                    labels.append(" ".join(label_parts))

            max_concurrent_tracks = max(max_concurrent_tracks, len(frame_track_ids))

            annotated_frame = box_annotator.annotate(
                scene=frame.copy(),
                detections=detections,
            )
            if show_labels or show_track_ids:
                annotated_frame = label_annotator.annotate(
                    scene=annotated_frame,
                    detections=detections,
                    labels=labels,
                )

            if use_boundary_filter and valid_boundary_points:
                annotated_frame = _draw_boundary_overlay(
                    frame=annotated_frame,
                    boundary_points=boundary_points or [],
                )

            if save_output_video:
                if video_writer is None:
                    if resolved_output_video_path is None:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_output:
                            resolved_output_video_path = tmp_output.name

                    output_height, output_width = annotated_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        resolved_output_video_path,
                        fourcc,
                        fps,
                        (output_width, output_height),
                    )
                    if not video_writer.isOpened():
                        raise ValueError(
                            f"Unable to open video writer at: {resolved_output_video_path}"
                        )

                video_writer.write(annotated_frame)

            if fallback_frame_bgr is None:
                fallback_frame_bgr = annotated_frame.copy()
                fallback_frame_index = frame_index
                fallback_frame_raw_bgr = frame.copy()
                if use_boundary_filter and valid_boundary_points:
                    fallback_frame_raw_bgr = _draw_boundary_overlay(
                        frame=fallback_frame_raw_bgr,
                        boundary_points=boundary_points or [],
                        color_bgr=(255, 255, 255),
                    )

            if sample_frame_bgr is None and frame_detection_count > 0:
                sample_frame_bgr = annotated_frame.copy()
                sample_frame_index = frame_index
                sample_frame_raw_bgr = frame.copy()
                if use_boundary_filter and valid_boundary_points:
                    sample_frame_raw_bgr = _draw_boundary_overlay(
                        frame=sample_frame_raw_bgr,
                        boundary_points=boundary_points or [],
                        color_bgr=(255, 255, 255),
                    )

            if detections.class_id is not None:
                for cls_id in detections.class_id.tolist():
                    cls_id_int = int(cls_id)
                    cls_name = model.names.get(cls_id_int, str(cls_id_int))
                    effective_name = cls_name
                    if label_renames is not None:
                        mapped_name = label_renames.get(cls_name, "")
                        if mapped_name.strip():
                            effective_name = mapped_name.strip()
                    class_counts[effective_name] = class_counts.get(effective_name, 0) + 1

            frame_index += 1
    finally:
        cap.release()
        if video_writer is not None:
            video_writer.release()
        if cleanup_temp_video:
            os.remove(video_path)

    average_detections_per_frame = (
        total_detections / frame_index if frame_index else 0.0
    )

    if sample_frame_bgr is None:
        sample_frame_bgr = fallback_frame_bgr
        sample_frame_index = fallback_frame_index

    if sample_frame_raw_bgr is None:
        sample_frame_raw_bgr = fallback_frame_raw_bgr

    detection_heatmap_bgr = build_spatial_heatmap_overlay(
        points=detection_center_points,
        frame_shape=(sample_frame_raw_bgr.shape[0], sample_frame_raw_bgr.shape[1])
        if sample_frame_raw_bgr is not None
        else (0, 0),
        base_frame_bgr=sample_frame_raw_bgr,
    )

    per_class_unique_tracks = {
        class_name: len(track_ids)
        for class_name, track_ids in per_class_unique_track_ids.items()
    }

    per_second_records = []
    for second_index in sorted(per_second_detection_counts.keys()):
        per_second_records.append(
            {
                "second_index": second_index,
                "timestamp_sec": float(second_index),
                "bird_count": int(per_second_detection_counts[second_index]),
                "unique_ids_active": int(
                    len(per_second_active_ids.get(second_index, set()))
                ),
            }
        )

    per_id_records = []
    for track_id in sorted(per_id_metrics.keys()):
        id_record = per_id_metrics[track_id]
        first_frame = int(id_record["first_frame"])
        last_frame = int(id_record["last_frame"])
        duration_frames = max(0, last_frame - first_frame + 1)
        per_id_records.append(
            {
                "track_id": int(id_record["track_id"]),
                "class_label": str(id_record["class_label"]),
                "first_frame": first_frame,
                "last_frame": last_frame,
                "duration_frames": duration_frames,
                "duration_seconds": (duration_frames / fps) if fps > 0 else 0.0,
                "total_detections": int(id_record["total_detections"]),
            }
        )

    return {
        "model_filepath": model_filepath,
        "confidence_threshold": conf_threshold,
        "tracking_enabled": True,
        "tracking_algorithm": tracker_config,
        "use_boundary_filter": boundary_filter_applied,
        "overlap_threshold_pct": overlap_threshold_pct,
        "boundary_points_count": len(boundary_points or []),
        "filtered_out_detections": total_filtered_out,
        "boundary_filter_warning": boundary_filter_warning,
        "total_frames": frame_index,
        "video_fps": fps,
        "total_detections": total_detections,
        "total_unique_tracks": len(unique_track_ids),
        "per_class_unique_tracks": per_class_unique_tracks,
        "max_concurrent_tracks": max_concurrent_tracks,
        "average_detections_per_frame": average_detections_per_frame,
        "class_counts": class_counts,
        "per_frame_detection_counts": per_frame_counts,
        "csv_per_second_records": per_second_records,
        "csv_per_id_records": per_id_records,
        "detection_heatmap_bgr": detection_heatmap_bgr,
        "detection_heatmap_point_count": len(detection_center_points),
        "sample_frame_bgr": sample_frame_bgr,
        "sample_frame_index": sample_frame_index,
        "output_video_path": resolved_output_video_path if save_output_video else None,
    }
  
