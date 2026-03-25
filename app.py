import streamlit as st

from utils.extract_frame import extractFrames
from utils.annotation import annotate
import tempfile
import os 
from ui.sidebar import get_sidebar_inputs, render_annotation_sidebar
from ui.parameter import analysis_parameters


st.set_page_config(page_title="Platform Use Analyzer", layout="centered")
st.title("Platform Use Analysis")

upload_file, bound_type, number_points, color = get_sidebar_inputs()

if "cached_upload_token" not in st.session_state:
    st.session_state.cached_upload_token = None
if "cached_frame" not in st.session_state:
    st.session_state.cached_frame = None


if upload_file is not None:
    upload_token = f"{upload_file.name}:{upload_file.size}"

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

    analysis_parameters()