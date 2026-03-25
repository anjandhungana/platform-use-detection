import streamlit as st


def _sync_heatmap_with_pdf():
    if not st.session_state.get("pdf_choice", True):
        st.session_state["heatmap_choice"] = False


def _sync_boundary_with_video():
    if not st.session_state.get("video_choice", True):
        st.session_state["video_boundary_choice"] = False


def analysis_parameters():
    # Initialize session state defaults for keyed widgets
    if "pdf_choice" not in st.session_state:
        st.session_state["pdf_choice"] = True
    if "heatmap_choice" not in st.session_state:
        st.session_state["heatmap_choice"] = True
    if "video_choice" not in st.session_state:
        st.session_state["video_choice"] = True
    if "video_boundary_choice" not in st.session_state:
        st.session_state["video_boundary_choice"] = True
    
    col1, col2 = st.columns([1, 3])
    with col1:
        overlap_threshold = st.number_input("Overlap threshold (%): ", min_value=0, max_value=100, value=30)
    with col2:
        models = {
            0: 'YOLO11n',
            1: 'YOLO11s',
            2: 'YOLO11m',
            3: 'YOLO11l',
            4: 'YOLO11x',
        }
        model = st.segmented_control(
            "Model selection",
            options=models.keys(),
            default=2,
            format_func=lambda option: models[option],
        )

    st.markdown("##### PDF report options")
    col1, col2 = st.columns(2, vertical_alignment="center")

    with col1:
        pdf_choice = st.checkbox(
            'Save report as PDF',
            key="pdf_choice",
            on_change=_sync_heatmap_with_pdf,
        )

    with col2:
        heatmap_choice = st.checkbox(
            'Report detection heatmap',
            key="heatmap_choice",
            disabled=not st.session_state.get("pdf_choice", True),
        )

    st.markdown("##### Video report options")
    col3, col4 = st.columns(2, vertical_alignment="center")
    with col3:
        video_choice = st.checkbox(
            'Generate video report',
            key="video_choice",
            on_change=_sync_boundary_with_video,
        )
    with col4:
        video_boundary_choice = st.checkbox(
            'Use boundary in video',
            key="video_boundary_choice",
            disabled=not st.session_state.get("video_choice", True),
        )

    return overlap_threshold