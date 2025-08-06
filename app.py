import streamlit as st
from ultralytics import YOLO
from PIL import Image
import json
import io 
import os
import zipfile
import numpy as np

# Cache the model load to avoid reloading on each inference
model = YOLO(r"C:\Users\Sivak\OneDrive\Desktop\Data Science PB\Deep Learning\Yolo_Streamlit_App\detection_model.pt")

# Custom CSS for coloring and styling
st.markdown("""
    <style>
    .main-title {
       
        font-size: 36px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    /* Subtitle */
    .subtitle {
        color: #077b8a;
        font-size: 20px;
        text-align: center;
        margin-bottom: 25px;
    }
    /* Colored count labels */
    .regular-hole {
        color: #fbb13c;
        font-weight: 700;
        font-size: 18px;
    }
    .threaded-hole {
        color: #DC143C;
        font-weight: 700;
        font-size: 18px;
    }
    /* Download button customization */
    div.stDownloadButton > button {
        background-color: #3abea6;
        color: white;
        font-weight: 600;
        padding: 10px 25px;
        border-radius: 8px;
    }
    div.stDownloadButton > button:hover {
        background-color: #fbb13c;
        color: #000000;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'> Metal Plate Hole Detection (YOLOv8) </div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload images to detect <span class='regular-hole'>Regular holes</span> and <span class='threaded-hole'>Threaded holes</span></div>", unsafe_allow_html=True)


uploaded_files = st.file_uploader("Choose image files...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)


if uploaded_files is not None:
    # In-memory bytes buffer for zip archive
    zip_buffer = io.BytesIO()
    
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file)
        # If you want to hide images, do not call st.image()
        # st.image(image, caption=f"Uploaded Image: {uploaded_file.name}", use_column_width=True)
    
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            image_np = np.array(image)  # PIL Image to numpy array
            res = model.predict(image_np, device='cpu', conf=0.25)
    
            regular_holes = []
            threaded_holes = []
            for result in res:
                for box in result.boxes:
                    class_id = box.cls.item()
                    coords = box.xyxy.tolist()[0]  # [x1, y1, x2, y2]
                    if class_id == 0:
                        regular_holes.append(coords)
                    elif class_id == 1:
                        threaded_holes.append(coords)
    
            # Display counts with color-coded labels using columns
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"<span class='regular-hole'> Regular holes detected in <b>{uploaded_file.name}</b>:</span> {len(regular_holes)}", unsafe_allow_html=True)
            with col2:
                st.markdown(f"<span class='threaded-hole'> Threaded holes detected in <b>{uploaded_file.name}</b>:</span> {len(threaded_holes)}", unsafe_allow_html=True)
    
            # Prepare JSON output
            output = {
                "regular_holes_count": len(regular_holes),
                "regular_holes": regular_holes,
                "threaded_holes_count": len(threaded_holes),
                "threaded_holes": threaded_holes,
            }
    
            # Create JSON string
            json_str = json.dumps(output, indent=2)
    
            # Use image file name prefix with .json extension
            json_filename = os.path.splitext(uploaded_file.name)[0] + '_detection.json'
    
            # Add JSON file as bytes to zip archive
            zip_file.writestr(json_filename, json_str)
    
    # Important: seek to start so BytesIO can be read from the beginning   
    zip_buffer.seek(0)
    
    # Single download button for the zip containing all JSON files
    st.download_button(
        label="Download All JSON Detections as ZIP",
        data=zip_buffer,
        file_name="metal_holes_detections.zip",
        mime="application/zip"
    )
else:
    st.info("Please upload one or more image files of a metal plate to detect holes.")
