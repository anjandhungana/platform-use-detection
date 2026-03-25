import streamlit as st

from utils.extract_frame import extractFrames
from utils.annotation import annotate
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