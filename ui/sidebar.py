import streamlit as st


def uploader():
    upload_file = st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov"])
    return upload_file


def boundary():
    options = ['points']
    number_points = None

    bound_type = st.sidebar.selectbox(
        'Choose boundary selection method',
        options
    )

    if bound_type == "points":
        number_points = st.sidebar.slider(label='Number of points', min_value=4, max_value=8)

    color = st.sidebar.color_picker(label="Boundary", value="#FFF")

    return bound_type, number_points, color


def get_sidebar_inputs():
    upload_file = uploader()
    bound_type, number_points, color = boundary()
    return upload_file, bound_type, number_points, color


def render_annotation_sidebar(annotation_data):
    if not annotation_data:
        return

    bound_type = annotation_data.get("bound_type")
    annotation_value = annotation_data.get("annotation_value")

    if bound_type == "points":
        selected_points = annotation_data.get("selected_points", [])
        max_points = annotation_data.get("max_points", 4)
        max_reached = annotation_data.get("max_reached", False)

        if st.sidebar.button("Clear selected points"):
            st.session_state.selected_points = []
            st.session_state.last_click = None
            selected_points = []
            max_reached = False
            
            if annotation_value is not None:
                st.session_state.suppress_click_once = (
                    annotation_value["x"],
                    annotation_value["y"],
                )

            # Redraw main image after state reset so cleared annotations disappear.
            st.rerun()

        if max_reached:
            st.sidebar.success("Maximum number of points reached.")
       
        
        st.sidebar.write(f"Selected points: {len(selected_points)}/{max_points}")

        point_rows = [
            {
                "Point": idx + 1,
                "x": selected_point["x"],
                "y": selected_point["y"],
            }
            for idx, selected_point in enumerate(selected_points)
        ]
        if point_rows:
            table_rows_html = "".join(
                f"<tr><td>{row['Point']}</td><td>{row['x']}</td><td>{row['y']}</td></tr>"
                for row in point_rows
            )
            table_html = (
                "<table style='width:100%; border-collapse:collapse; text-align:center;'>"
                "<thead><tr><th>Point</th><th>x</th><th>y</th></tr></thead>"
                f"<tbody>{table_rows_html}</tbody></table>"
            )
            st.sidebar.markdown(table_html, unsafe_allow_html=True)


        

    elif annotation_value is not None:
        st.sidebar.write(
            "Rectangle anchor coordinates:",
            {"x": annotation_value["x"], "y": annotation_value["y"]},
        )

