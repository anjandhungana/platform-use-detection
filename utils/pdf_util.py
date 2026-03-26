from __future__ import annotations

import base64
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _build_timeline_svg(per_frame_counts: list[int]) -> str | None:
	if not per_frame_counts:
		return None

	width = 820
	height = 180
	padding_x = 32
	padding_y = 20
	plot_width = width - (padding_x * 2)
	plot_height = height - (padding_y * 2)

	max_count = max(per_frame_counts)
	safe_max = max(max_count, 1)
	n = len(per_frame_counts)

	if n == 1:
		points = [(padding_x + plot_width / 2.0, padding_y + plot_height / 2.0)]
	else:
		points = []
		for idx, value in enumerate(per_frame_counts):
			x = padding_x + (plot_width * idx / (n - 1))
			y = padding_y + plot_height - ((value / safe_max) * plot_height)
			points.append((x, y))

	polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)

	return f"""
<svg viewBox=\"0 0 {width} {height}\" xmlns=\"http://www.w3.org/2000/svg\" role=\"img\" aria-label=\"Detection timeline\">
  <rect x=\"0\" y=\"0\" width=\"{width}\" height=\"{height}\" fill=\"#f7fafc\" rx=\"10\"/>
  <line x1=\"{padding_x}\" y1=\"{padding_y + plot_height}\" x2=\"{padding_x + plot_width}\" y2=\"{padding_y + plot_height}\" stroke=\"#94a3b8\" stroke-width=\"1\"/>
  <line x1=\"{padding_x}\" y1=\"{padding_y}\" x2=\"{padding_x}\" y2=\"{padding_y + plot_height}\" stroke=\"#94a3b8\" stroke-width=\"1\"/>
  <polyline fill=\"none\" stroke=\"#0f766e\" stroke-width=\"3\" stroke-linecap=\"round\" stroke-linejoin=\"round\" points=\"{polyline}\"/>
  <text x=\"{padding_x}\" y=\"14\" fill=\"#0f172a\" font-size=\"12\">Max detections: {max_count}</text>
  <text x=\"{padding_x}\" y=\"{height - 6}\" fill=\"#475569\" font-size=\"11\">Frame 0</text>
  <text x=\"{padding_x + plot_width - 60}\" y=\"{height - 6}\" fill=\"#475569\" font-size=\"11\">Frame {n - 1}</text>
</svg>
""".strip()


def _encode_frame_png(frame_bgr: np.ndarray | None) -> str | None:
	if frame_bgr is None:
		return None

	ok, encoded = cv2.imencode(".png", frame_bgr)
	if not ok:
		return None

	png_bytes = encoded.tobytes()
	b64 = base64.b64encode(png_bytes).decode("ascii")
	return f"data:image/png;base64,{b64}"


def generate_pdf_report(
	inference_summary: dict[str, Any],
	source_video_name: str,
	model_name: str | None = None,
	include_heatmap: bool = True,
	output_pdf_path: str | None = None,
) -> str:
	try:
		from jinja2 import Environment, FileSystemLoader, select_autoescape
		from weasyprint import CSS, HTML
	except Exception as exc:
		raise RuntimeError(
			"Missing PDF dependencies. Install with: pip install jinja2 weasyprint"
		) from exc

	template_dir = (
		Path(__file__).resolve().parent.parent / "assets" / "report_templates"
	)

	env = Environment(
		loader=FileSystemLoader(str(template_dir)),
		autoescape=select_autoescape(["html", "xml"]),
	)
	template = env.get_template("video_report.html")

	stats = {
		"Total Frames": inference_summary.get("total_frames", 0),
		"FPS": f"{float(inference_summary.get('video_fps', 0.0)):.2f}",
		"Total Detections": inference_summary.get("total_detections", 0),
		"Average Detections / Frame": f"{float(inference_summary.get('average_detections_per_frame', 0.0)):.2f}",
		"Unique Tracks": inference_summary.get("total_unique_tracks", 0),
		"Boundary Filter Enabled": "Yes" if inference_summary.get("use_boundary_filter", False) else "No",
	}

	if inference_summary.get("use_boundary_filter", False):
		stats["Boundary Overlap Threshold (%)"] = (
			f"{float(inference_summary.get('overlap_threshold_pct', 0.0)):.2f}"
		)

	sample_frame_data_uri = _encode_frame_png(inference_summary.get("sample_frame_bgr"))
	detection_heatmap_data_uri = None
	timeline_svg = None
	if include_heatmap:
		detection_heatmap_data_uri = _encode_frame_png(
			inference_summary.get("detection_heatmap_bgr")
		)
		timeline_svg = _build_timeline_svg(
			inference_summary.get("per_frame_detection_counts", []) or []
		)

	rendered_html = template.render(
		report_title="Platform Use Analysis Report",
		source_video_name=source_video_name,
		generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
		model_name=model_name or inference_summary.get("model_filepath", ""),
		confidence_threshold=float(inference_summary.get("confidence_threshold", 0.0)),
		stats=stats,
		sample_frame_data_uri=sample_frame_data_uri,
		sample_frame_index=inference_summary.get("sample_frame_index"),
		detection_heatmap_data_uri=detection_heatmap_data_uri,
		detection_heatmap_point_count=int(
			inference_summary.get("detection_heatmap_point_count", 0)
		),
		include_heatmap=include_heatmap,
		timeline_svg=timeline_svg,
	)

	if output_pdf_path is None:
		with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
			output_pdf_path = tmp_file.name

	stylesheets = []
	css_path = template_dir / "video_report.css"
	if css_path.is_file():
		stylesheets.append(CSS(filename=str(css_path)))

	HTML(string=rendered_html, base_url=str(template_dir)).write_pdf(
		output_pdf_path,
		stylesheets=stylesheets,
	)
	return output_pdf_path
