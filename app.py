import streamlit as st
import cv2
from ultralytics import YOLO

from utils.extract_frame import extractFrames
from utils.annotation import annotate
from utils.run_inference import run_inference
from utils.pdf_util import generate_pdf_report
from utils.csv_report import generate_csv_reports
import tempfile
import os 
from ui.sidebar import get_sidebar_inputs, render_annotation_sidebar
from ui.parameter import analysis_parameters
import base64


def _get_background_css():
    """Load background image and create CSS with base64 encoding."""
    try:
        bg_path = os.path.join(os.path.dirname(__file__), "assets", "background.png")
        with open(bg_path, "rb") as img_file:
            img_base64 = base64.b64encode(img_file.read()).decode()
        return f"""
        <style>
            .stApp {{
                background-image: url("data:image/png;base64,{img_base64}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            .stApp::before {{
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: linear-gradient(to bottom, rgba(0, 0, 0, 0.98), rgba(0, 0, 0, 0.87));
                pointer-events: none;
                z-index: 1;
            }}
            .stApp > * {{
                position: relative;
                z-index: 2;
            }}
        </style>
        """
    except Exception as e:
        print(f"Warning: Could not load background image: {e}")
        return ""


def _default_display_label(class_name: str) -> str:
    name = class_name.strip()
    if not name:
        return class_name

    if name.lower() == "chickens":
        return "chicken"

    if len(name) > 3 and name.lower().endswith("s"):
        return name[:-1]

    return name


def _get_model_class_names(model_filepath: str) -> list[str]:
    if "model_class_names_cache" not in st.session_state:
        st.session_state.model_class_names_cache = {}

    cache = st.session_state.model_class_names_cache
    if model_filepath in cache:
        return cache[model_filepath]

    model = YOLO(model_filepath)
    names = model.names
    ordered_names = [str(names[idx]) for idx in sorted(names.keys())]
    cache[model_filepath] = ordered_names
    return ordered_names


st.set_page_config(page_title="Platform Use Analyzer", layout="centered")

# Add background image with gradient overlay
st.markdown(_get_background_css(), unsafe_allow_html=True)

st.title("Platform Use Analysis")

# Add CSS for the upload prompt box
st.markdown("""
    <style>
        .upload-prompt-box {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 300px;
            margin: 40px auto;
            padding: 40px;
            background-color: rgba(255, 255, 255, 0.05);
            border: 2px solid rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            backdrop-filter: blur(10px);
            text-align: center;
        }
        .upload-prompt-text {
            font-size: 32px;
            font-weight: bold;
            color: rgba(255, 255, 255, 0.8);
            letter-spacing: 0.5px;
        }
    </style>
    """, unsafe_allow_html=True)

upload_file, bound_type, number_points, color = get_sidebar_inputs()

MODEL_FILEPATHS = {
    0: os.path.join("assets", "models", "yolo11nA1.pt"),
    1: os.path.join("assets", "models", "yolo11sA1.pt"),
    2: os.path.join("assets", "models", "yolo11mA1.pt"),
    3: os.path.join("assets", "models", "yolo11lA1.pt"),
    4: os.path.join("assets", "models", "yolo11xA1.pt"),
}

MODEL_DISPLAY_NAMES = {
    0: "YOLO11n",
    1: "YOLO11s",
    2: "YOLO11m",
    3: "YOLO11l",
    4: "YOLO11x",
}

if "cached_upload_token" not in st.session_state:
    st.session_state.cached_upload_token = None
if "cached_frame" not in st.session_state:
    st.session_state.cached_frame = None
if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None
if "analysis_upload_token" not in st.session_state:
    st.session_state.analysis_upload_token = None
if "analysis_video_bytes" not in st.session_state:
    st.session_state.analysis_video_bytes = None
if "analysis_video_filename" not in st.session_state:
    st.session_state.analysis_video_filename = None
if "analysis_pdf_bytes" not in st.session_state:
    st.session_state.analysis_pdf_bytes = None
if "analysis_pdf_filename" not in st.session_state:
    st.session_state.analysis_pdf_filename = None
if "analysis_csv_timeline_bytes" not in st.session_state:
    st.session_state.analysis_csv_timeline_bytes = None
if "analysis_csv_timeline_filename" not in st.session_state:
    st.session_state.analysis_csv_timeline_filename = None
if "analysis_csv_id_stats_bytes" not in st.session_state:
    st.session_state.analysis_csv_id_stats_bytes = None
if "analysis_csv_id_stats_filename" not in st.session_state:
    st.session_state.analysis_csv_id_stats_filename = None


if upload_file is not None:
    upload_token = f"{upload_file.name}:{upload_file.size}"

    # Reset stale analysis outputs when a new file is selected.
    if st.session_state.analysis_upload_token != upload_token:
        st.session_state.analysis_result = None
        st.session_state.analysis_upload_token = None
        st.session_state.analysis_video_bytes = None
        st.session_state.analysis_video_filename = None
        st.session_state.analysis_pdf_bytes = None
        st.session_state.analysis_pdf_filename = None
        st.session_state.analysis_csv_timeline_bytes = None
        st.session_state.analysis_csv_timeline_filename = None
        st.session_state.analysis_csv_id_stats_bytes = None
        st.session_state.analysis_csv_id_stats_filename = None

    if (
        st.session_state.cached_upload_token != upload_token
        or st.session_state.cached_frame is None
    ):
        with tempfile.NamedTemporaryFile(delete=False) as tfile:
            tfile.write(upload_file.getvalue())
            temp_video_path = tfile.name

        extracted_frame = extractFrames(temp_video_path)
        os.remove(temp_video_path)

        if extracted_frame is not None:
            st.session_state.cached_frame = extracted_frame
            st.session_state.cached_upload_token = upload_token
        else:
            st.session_state.cached_frame = None
            st.error("Could not extract a frame from this video. Please try another file.")

    annotation_data = None
    img = st.session_state.cached_frame
    if img is not None:
        annotation_data = annotate(
            img,
            bound_type,
            number_points=number_points,
            upload_token=upload_token,
            boundary_color=color,
        )
        render_annotation_sidebar(annotation_data)

    (
        overlap_threshold,
        confidence_threshold,
        selected_model,
        pdf_choice,
        heatmap_choice,
        video_choice,
        video_label_choice,
        video_id_choice,
        video_boundary_choice,
        video_rename_label_choice,
        csv_choice,
    ) = analysis_parameters()

    model_relative_path = MODEL_FILEPATHS.get(selected_model)
    model_display_name = MODEL_DISPLAY_NAMES.get(selected_model, "Unknown")
    if model_relative_path is None:
        st.error("Invalid model selection.")
        st.stop()

    model_filepath = os.path.join(os.path.dirname(__file__), model_relative_path)

    label_renames: dict[str, str] = {}
    try:
        model_class_names = _get_model_class_names(model_filepath)
    except Exception as exc:
        st.warning(f"Could not load model class names for renaming: {exc}")
        model_class_names = []

    if model_class_names and video_label_choice and video_rename_label_choice:
        st.markdown("##### Label rename")
        old_col, new_col = st.columns([1, 2])
        with old_col:
            st.caption("Old label")
        with new_col:
            st.caption("New label")

        for class_name in model_class_names:
            key = f"label_rename_{selected_model}_{class_name}"
            left_col, right_col = st.columns([1, 2], vertical_alignment="center")
            with left_col:
                st.write(class_name)
            with right_col:
                rename_value = st.text_input(
                    "New label",
                    value=st.session_state.get(key, _default_display_label(class_name)),
                    placeholder="Enter replacement label",
                    key=key,
                    label_visibility="collapsed",
                )
            label_renames[class_name] = rename_value

    with st.container(horizontal_alignment='center'):
        analyze_clicked = st.button('Analyze Video')

    if analyze_clicked:
        normalized_conf_threshold = confidence_threshold / 100.0
        boundary_points = []
        if annotation_data:
            boundary_points = annotation_data.get("selected_points", [])

        with st.spinner("Running inference..."):
            try:
                inference_summary = run_inference(
                    video_input=upload_file,
                    model_filepath=model_filepath,
                    conf_threshold=normalized_conf_threshold,
                    show_labels=video_label_choice,
                    show_track_ids=video_id_choice,
                    label_renames=label_renames,
                    save_output_video=video_choice,
                    use_boundary_filter=video_boundary_choice,
                    boundary_points=boundary_points,
                    overlap_threshold_pct=overlap_threshold,
                )
            except Exception as exc:
                st.error(f"Inference failed: {exc}")
            else:
                st.session_state.analysis_result = inference_summary
                st.session_state.analysis_upload_token = upload_token

                output_name = os.path.splitext(upload_file.name)[0]

                output_video_path = inference_summary.get("output_video_path")
                st.session_state.analysis_video_bytes = None
                st.session_state.analysis_video_filename = None
                if output_video_path and os.path.isfile(output_video_path):
                    with open(output_video_path, "rb") as output_video_file:
                        st.session_state.analysis_video_bytes = output_video_file.read()
                    st.session_state.analysis_video_filename = f"{output_name}_report.mp4"

                st.session_state.analysis_pdf_bytes = None
                st.session_state.analysis_pdf_filename = None
                st.session_state.analysis_csv_timeline_bytes = None
                st.session_state.analysis_csv_timeline_filename = None
                st.session_state.analysis_csv_id_stats_bytes = None
                st.session_state.analysis_csv_id_stats_filename = None
                if pdf_choice:
                    try:
                        output_pdf_path = generate_pdf_report(
                            inference_summary=inference_summary,
                            source_video_name=upload_file.name,
                            model_name=model_display_name,
                            include_heatmap=heatmap_choice,
                        )
                    except Exception as exc:
                        st.warning(f"PDF report generation failed: {exc}")
                    else:
                        if os.path.isfile(output_pdf_path):
                            with open(output_pdf_path, "rb") as output_pdf_file:
                                st.session_state.analysis_pdf_bytes = output_pdf_file.read()
                            st.session_state.analysis_pdf_filename = f"{output_name}_report.pdf"

                if csv_choice:
                    try:
                        csv_reports = generate_csv_reports(
                            inference_summary=inference_summary,
                            source_video_name=upload_file.name,
                        )
                    except Exception as exc:
                        st.warning(f"CSV report generation failed: {exc}")
                    else:
                        timeline_path = csv_reports.get("timeline_path")
                        if timeline_path and os.path.isfile(timeline_path):
                            with open(timeline_path, "rb") as timeline_file:
                                st.session_state.analysis_csv_timeline_bytes = (
                                    timeline_file.read()
                                )
                            st.session_state.analysis_csv_timeline_filename = csv_reports.get(
                                "timeline_filename"
                            )

                        id_stats_path = csv_reports.get("id_stats_path")
                        if id_stats_path and os.path.isfile(id_stats_path):
                            with open(id_stats_path, "rb") as id_stats_file:
                                st.session_state.analysis_csv_id_stats_bytes = (
                                    id_stats_file.read()
                                )
                            st.session_state.analysis_csv_id_stats_filename = csv_reports.get(
                                "id_stats_filename"
                            )

    inference_summary = st.session_state.analysis_result
    if inference_summary is not None:
        boundary_filter_warning = inference_summary.get("boundary_filter_warning")
        if boundary_filter_warning:
            st.warning(boundary_filter_warning)

        st.success("Inference completed.")

        sample_frame_bgr = inference_summary.get("sample_frame_bgr")
        sample_frame_index = inference_summary.get("sample_frame_index")
        if sample_frame_bgr is not None:
            sample_frame_rgb = cv2.cvtColor(sample_frame_bgr, cv2.COLOR_BGR2RGB)
            st.write("Inference sample image")
            st.image(
                sample_frame_rgb,
                caption=f"Annotated frame #{sample_frame_index}",
                use_container_width=True,
            )

            heatmap_bgr = inference_summary.get("detection_heatmap_bgr")
            if heatmap_bgr is not None:
                heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
                st.write("Overlayed heatmap")
                st.image(
                    heatmap_rgb,
                    caption=f"Heatmap overlay for frame #{sample_frame_index}",
                    use_container_width=True,
                )

        video_bytes = st.session_state.analysis_video_bytes
        video_filename = st.session_state.analysis_video_filename
        pdf_bytes = st.session_state.analysis_pdf_bytes
        pdf_filename = st.session_state.analysis_pdf_filename
        csv_timeline_bytes = st.session_state.analysis_csv_timeline_bytes
        csv_timeline_filename = st.session_state.analysis_csv_timeline_filename
        csv_id_stats_bytes = st.session_state.analysis_csv_id_stats_bytes
        csv_id_stats_filename = st.session_state.analysis_csv_id_stats_filename
        if (
            video_bytes is not None
            or pdf_bytes is not None
            or csv_timeline_bytes is not None
            or csv_id_stats_bytes is not None
        ):
            st.write("Report download")
            report_col1, report_col2 = st.columns(2)
            with report_col1:
                if video_bytes is not None and video_filename:
                    st.download_button(
                        "Download video report",
                        data=video_bytes,
                        file_name=video_filename,
                        mime="video/mp4",
                    )
            with report_col2:
                if pdf_bytes is not None and pdf_filename:
                    st.download_button(
                        "Download PDF report",
                        data=pdf_bytes,
                        file_name=pdf_filename,
                        mime="application/pdf",
                    )

            csv_col1, csv_col2 = st.columns(2)
            with csv_col1:
                if csv_timeline_bytes is not None and csv_timeline_filename:
                    st.download_button(
                        "Download timeline CSV",
                        data=csv_timeline_bytes,
                        file_name=csv_timeline_filename,
                        mime="text/csv",
                    )
            with csv_col2:
                if csv_id_stats_bytes is not None and csv_id_stats_filename:
                    st.download_button(
                        "Download ID stats CSV",
                        data=csv_id_stats_bytes,
                        file_name=csv_id_stats_filename,
                        mime="text/csv",
                    )

else:
    # Show upload prompt when no file is uploaded
    st.markdown(
        """
        <div class="upload-prompt-box">
            <div class="upload-prompt-text">Use upload button on sidebar to get started</div>
        </div>
        """,
        unsafe_allow_html=True
    )