import streamlit as st


def _sync_heatmap_with_pdf():
    if not st.session_state.get("pdf_choice", True):
        st.session_state["heatmap_choice"] = False


def _sync_boundary_with_video():
    if not st.session_state.get("video_choice", True):
        st.session_state["video_boundary_choice"] = False
        st.session_state["video_label_choice"] = False
        st.session_state["video_id_choice"] = False
        st.session_state["video_rename_label_choice"] = False


def _sync_label_with_video():
    if not st.session_state.get("video_choice", True):
        st.session_state["video_label_choice"] = False
    if not st.session_state.get("video_label_choice", False):
        st.session_state["video_rename_label_choice"] = False


def _sync_id_with_video():
    if not st.session_state.get("video_choice", True):
        st.session_state["video_id_choice"] = False


def _sync_rename_label_with_video():
    if not st.session_state.get("video_choice", True):
        st.session_state["video_rename_label_choice"] = False


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
    if "video_label_choice" not in st.session_state:
        st.session_state["video_label_choice"] = False
    if "video_id_choice" not in st.session_state:
        st.session_state["video_id_choice"] = True
    if "video_rename_label_choice" not in st.session_state:
        st.session_state["video_rename_label_choice"] = False
    if "csv_choice" not in st.session_state:
        st.session_state["csv_choice"] = False
    
    col1, col2 = st.columns(2)
    with col1:
        overlap_threshold = st.number_input("Overlap threshold (%): ", min_value=0, max_value=100, value=30)

    with col2:
        confidence_threshold = st.number_input(
            "Confidence threshold (%): ",
            min_value=0,
            max_value=100,
            value=25,
        )

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
    row1_col1, row1_col2 = st.columns(2, vertical_alignment="center")
    with row1_col1:
        video_choice = st.checkbox(
            'Generate video report',
            key="video_choice",
            on_change=_sync_boundary_with_video,
        )

    video_boundary_choice = st.session_state.get("video_boundary_choice", False)
    video_id_choice = st.session_state.get("video_id_choice", False)
    video_label_choice = st.session_state.get("video_label_choice", False)
    video_rename_label_choice = st.session_state.get(
        "video_rename_label_choice",
        False,
    )

    if video_choice:
        with row1_col2:
            video_boundary_choice = st.checkbox(
                'Show boundary',
                key="video_boundary_choice",
            )

        row2_col1, row2_col2 = st.columns(2, vertical_alignment="top")
        with row2_col1:
            st.write("")

        with row2_col2:
            video_id_choice = st.checkbox(
                'Show IDs',
                key="video_id_choice",
                on_change=_sync_id_with_video,
            )

            video_label_choice = st.checkbox(
                'Show labels',
                key="video_label_choice",
                on_change=_sync_label_with_video,
            )

            video_rename_label_choice = st.checkbox(
                'Rename labels',
                key="video_rename_label_choice",
                on_change=_sync_rename_label_with_video,
                disabled=not st.session_state.get("video_label_choice", False),
            )

    st.markdown("##### CSV report options")
    csv_choice = st.checkbox(
        'Export CSV reports',
        key="csv_choice",
    )

    return (
        overlap_threshold,
        confidence_threshold,
        model,
        pdf_choice,
        heatmap_choice,
        video_choice,
        video_label_choice,
        video_id_choice,
        video_boundary_choice,
        video_rename_label_choice,
        csv_choice,
    )