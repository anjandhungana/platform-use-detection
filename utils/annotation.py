from streamlit_image_coordinates import streamlit_image_coordinates
from streamlit_js_eval import streamlit_js_eval
import streamlit as st
import cv2
import numpy as np


def _ensure_state():
    if "selected_points" not in st.session_state:
        st.session_state.selected_points = []
    if "last_click" not in st.session_state:
        st.session_state.last_click = None
    if "prev_mode" not in st.session_state:
        st.session_state.prev_mode = None
    if "prev_max_points" not in st.session_state:
        st.session_state.prev_max_points = None
    if "prev_upload_token" not in st.session_state:
        st.session_state.prev_upload_token = None
    if "suppress_click_once" not in st.session_state:
        st.session_state.suppress_click_once = None
    if "viewport_width" not in st.session_state:
        st.session_state.viewport_width = None


def _map_click_to_original(img, component_key="annotation_click_canvas"):
    original_height, original_width = img.shape[:2]

    if st.session_state.viewport_width is None:
        viewport_width = streamlit_js_eval(
            js_expressions='window.innerWidth',
            key='WIDTH',
            want_output=True,
        )
        if viewport_width:
            st.session_state.viewport_width = int(viewport_width)

    display_width = st.session_state.viewport_width

    if not display_width:
        # Fallback width to avoid repeated JS probing that can cause UI flashes.
        display_width = min(original_width, 1000)

    # Avoid invalid widths and keep scaling deterministic.
    display_width = int(max(1, min(display_width, original_width)))
    display_height = (original_height * display_width) / original_width

    clicked = streamlit_image_coordinates(
        img,
        width=display_width,
        key=component_key,
    )

    if clicked:
        scale_x = original_width / display_width
        scale_y = original_height / display_height

        orig_x = clicked["x"] * scale_x
        orig_y = clicked["y"] * scale_y

        mapped = {
            "x": int(round(orig_x)),
            "y": int(round(orig_y)),
            "display_x": clicked["x"],
            "display_y": clicked["y"],
            "display_width": display_width,
            "display_height": int(round(display_height)),
            "original_width": original_width,
            "original_height": original_height,
        }

        return mapped

    return None


def _hex_to_bgr(color_hex):
    if not isinstance(color_hex, str):
        return (255, 255, 255)

    color = color_hex.strip().lstrip("#")
    if len(color) == 3:
        color = "".join(ch * 2 for ch in color)
    if len(color) != 6:
        return (255, 255, 255)

    try:
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
    except ValueError:
        return (255, 255, 255)

    # The image is rendered in RGB in Streamlit, so keep channel order RGB.
    return (r, g, b)


def _render_annotation_overlay(
    img,
    bound_type,
    boundary_color,
    close_polygon=False,
    fill_polygon=False,
):
    annotated_img = img.copy()
    color_bgr = _hex_to_bgr(boundary_color)

    if bound_type == "points":
        points = [
            (int(point["x"]), int(point["y"]))
            for point in st.session_state.selected_points
        ]

        if fill_polygon and len(points) >= 3:
            # Draw transparent fill on top of frame but beneath outline/markers.
            polygon_overlay = annotated_img.copy()
            cv2.fillPoly(
                polygon_overlay,
                [np.array(points, dtype=np.int32)],
                color_bgr,
            )
            annotated_img = cv2.addWeighted(polygon_overlay, 0.45, annotated_img, 0.55, 0)

        for idx in range(1, len(points)):
            cv2.line(
                annotated_img,
                points[idx - 1],
                points[idx],
                color_bgr,
                2,
            )

        if close_polygon and len(points) >= 3:
            cv2.line(
                annotated_img,
                points[-1],
                points[0],
                color_bgr,
                2,
            )

        for point in st.session_state.selected_points:
            center = (int(point["x"]), int(point["y"]))
            cv2.ellipse(
                annotated_img,
                center,
                (10, 10),
                0,
                0,
                360,
                color_bgr,
                2,
            )

    return annotated_img


def annotate(
    img,
    bound_type,
    number_points=None,
    upload_token=None,
    boundary_color="#FFF",
    overlay_enabled=False,
):
    _ensure_state()

    max_points = number_points or 4

    if (
        st.session_state.prev_upload_token != upload_token
        or st.session_state.prev_mode != bound_type
        or st.session_state.prev_max_points != number_points
    ):
        st.session_state.selected_points = []
        st.session_state.last_click = None

    st.session_state.prev_upload_token = upload_token
    st.session_state.prev_mode = bound_type
    st.session_state.prev_max_points = number_points

    should_close_polygon = (
        bound_type == "points"
        and len(st.session_state.selected_points) >= max_points
    )
    should_fill_polygon = should_close_polygon and overlay_enabled
    component_key = (
        f"annotation_click_canvas_{bound_type}_{boundary_color}_"
        f"{overlay_enabled}"
    )
    img_with_overlay = _render_annotation_overlay(
        img,
        bound_type,
        boundary_color,
        close_polygon=should_close_polygon,
        fill_polygon=should_fill_polygon,
    )
    annotation_value = _map_click_to_original(img_with_overlay, component_key=component_key)
    result = {
        "bound_type": bound_type,
        "annotation_value": annotation_value,
        "selected_points": st.session_state.selected_points,
        "max_points": max_points,
        "max_reached": False,
        "overlay_enabled": overlay_enabled,
    }

    if bound_type == "points":
        if annotation_value is not None:
            current_click = (annotation_value["x"], annotation_value["y"])
            is_suppressed_click = current_click == st.session_state.suppress_click_once
            is_initial_seed_click = (
                st.session_state.last_click is None
                and len(st.session_state.selected_points) == 0
                and current_click == (0, 0)
            )

            if is_suppressed_click:
                st.session_state.last_click = current_click
                st.session_state.suppress_click_once = None
                result["selected_points"] = st.session_state.selected_points
                result["max_points"] = max_points
                result["max_reached"] = len(st.session_state.selected_points) >= max_points
                return result

            if (
                not is_initial_seed_click
                and current_click != st.session_state.last_click
                and len(st.session_state.selected_points) < max_points
            ):
                st.session_state.selected_points.append(
                    {"x": annotation_value["x"], "y": annotation_value["y"]}
                )
                # Record this click before rerun so it is not re-added on the next pass.
                st.session_state.last_click = current_click
                # This component reports clicks on the following script pass.
                # Trigger one controlled rerun so markers/counter stay in sync.
                st.rerun()

            st.session_state.last_click = current_click

        result["selected_points"] = st.session_state.selected_points
        result["max_points"] = max_points
        result["max_reached"] = len(st.session_state.selected_points) >= max_points

    return result
    



