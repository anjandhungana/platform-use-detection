# Platform Use Analyzer (Streamlit)

Platform Use Analyzer is a Streamlit application for video-based bird detection and tracking using YOLO11 models. It supports polygon boundary filtering, annotated video export, PDF reporting, and CSV report generation.

## Features

- Upload video files (.mp4, .mov)
- Select from YOLO11n/s/m/l/x model variants
- Draw a polygon boundary on a frame to restrict detections
- Configure confidence and overlap thresholds
- Annotated video report export
- PDF report export with optional heatmap and timeline visualization
- CSV exports for timeline and per-ID statistics
- Optional label renaming in video overlays

## Project Structure

- app.py: Main Streamlit application
- ui/sidebar.py: Upload and boundary-related sidebar UI
- ui/parameter.py: Analysis controls and export options
- utils/run_inference.py: Detection, tracking, filtering, and summary generation
- utils/csv_report.py: Timeline and ID-stats CSV writers
- utils/pdf_util.py: PDF report rendering via Jinja2 + WeasyPrint
- utils/annotation.py: Click-based boundary point annotation
- assets/models/: YOLO model weights
- assets/report_templates/: PDF HTML/CSS templates
- assets/sample_vids/: Sample videos

## Requirements

Python 3.10+ is recommended.

Python packages used by this project:

- streamlit
- ultralytics
- supervision
- opencv-python
- numpy
- jinja2
- weasyprint
- streamlit-image-coordinates
- streamlit-js-eval

## Setup

1. Create and activate a virtual environment.

2. Install dependencies:

```bash
pip install streamlit ultralytics supervision opencv-python numpy jinja2 weasyprint streamlit-image-coordinates streamlit-js-eval
```

3. Ensure YOLO model weights are present in assets/models/:

- yolo11nA1.pt
- yolo11sA1.pt
- yolo11mA1.pt
- yolo11lA1.pt
- yolo11xA1.pt

## Run

From the project root:

```bash
streamlit run app.py
```

Then open the local URL shown by Streamlit in your browser.

## Workflow

1. Upload a video from the sidebar.
2. Select boundary points on the preview frame.
3. Configure analysis options:
   - Overlap threshold and confidence threshold
   - Model selection
   - Video/PDF/CSV export options
4. Click Analyze Video.
5. Download generated outputs.

## Output Reports

### Video report

MP4 with tracked detections and optional overlays (boundary, IDs, labels).

### PDF report

Includes summary metrics and optional heatmap/timeline sections.

### CSV reports

1) Timeline CSV (<video_name>_timeline.csv)

Columns:
- second_index
- timestamp_sec
- unique_bird_ids

2) ID stats CSV (<video_name>_id_stats.csv)

Columns:
- track_id
- class_label
- first_frame
- last_frame
- duration_frames
- duration_seconds (rounded to 2 decimals)

## Notes

- If boundary filtering is enabled with fewer than 3 points, inference falls back to no boundary filtering.
- PDF generation depends on WeasyPrint. On some Linux systems, additional system libraries may be required.
