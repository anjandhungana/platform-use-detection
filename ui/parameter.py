import streamlit as st

def analysis_parameters():
    col1, col2 = st.columns([1,3])
    with col1:
     overlap_threshold = st.number_input("Overlap threshold (%): ", min_value=0, max_value=100, value=30)
    with col2:
       models = {
          0:'YOLO11n', 1:'YOLO11s', 2:'YOLO11m', 3:'YOLO11l', 4:'YOLO11x',}
       model = st.segmented_control(
          "Model selection",
          options = models.keys(),
          default = 2,
          format_func = lambda option: models[option], 
       )

    col1, col2, col3 = st.columns(3, vertical_alignment="center")
    
    with col1:
        pdf_choice = st.checkbox('Save report as PDF', value=True)
    
    with col2:
        heatmap_choice = st.checkbox('Report detection heatmap', value=True)
    
    with col3:
        video_choice = st.checkbox('Generate video report', value=True)
    

    return overlap_threshold