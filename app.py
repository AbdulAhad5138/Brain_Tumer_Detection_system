import streamlit as st
from ultralytics import YOLO
import PIL.Image
import numpy as np
import cv2
import pandas as pd
import io

# --- Page Config ---
st.set_page_config(
    page_title="NeuroScan Pro - Brain Tumor Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom Styling (Glassmorphism & Neon) ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #f8fafc !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Upload Box */
    .stFileUploader {
        background-color: rgba(30, 41, 59, 0.5);
        border: 2px dashed #94a3b8;
        border-radius: 12px;
        padding: 20px;
    }
    
    /* Metrics */
    div[data-testid="stMetricValue"] {
        font-size: 24px;
        color: #38bdf8;
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #0f172a;
        border-right: 1px solid #334155;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #3b82f6 0%, #2563eb 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    </style>
""", unsafe_allow_html=True)

# --- Header ---
col_header_1, col_header_2 = st.columns([1, 4])
with col_header_1:
    st.markdown("# 🧠")
with col_header_2:
    st.title("NeuroScan Pro")
    st.markdown("### AI-Powered Brain Tumor Segmentation System")

st.divider()

# --- Model Loading ---
@st.cache_resource
def load_model():
    try:
        model = YOLO('best.pt')
        return model
    except Exception as e:
        return None

try:
    model = load_model()
    if model is None:
        st.error("❌ Model 'best.pt' not found or corrupt over!")
        st.stop()
except Exception as e:
    st.error(f"❌ Critical Error: {str(e)}")
    st.stop()

# --- Sidebar Controls ---
with st.sidebar:
    st.subheader("⚙️ Detection Settings")
    
    conf_threshold = st.slider(
        "Confidence Threshold", 
        0.0, 1.0, 0.25, 
        help="Minimum probability to count as a detection."
    )
    
    iou_threshold = st.slider(
        "IOU Threshold", 
        0.0, 1.0, 0.45, 
        help="Intersection over Union threshold for NMS."
    )
    
    st.subheader("🎨 Visuals")
    mask_opacity = st.slider("Mask Opacity", 0.0, 1.0, 0.4)
    line_width = st.slider("Border Thickness", 1, 5, 2)
    
    st.info("ℹ️ **About**: This tool uses YOLOv8-Seg to precisely outline tumor regions in MRI scans.")

# --- Main Logic ---
uploaded_file = st.file_uploader("📂 Upload MRI Scan (JPG, PNG, JPEG)", type=['jpg', 'jpeg', 'png', 'bmp'])

if uploaded_file:
    # 1. Read Image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_cv2 = cv2.imdecode(file_bytes, 1) # BGR
    image_rgb = cv2.cvtColor(image_cv2, cv2.COLOR_BGR2RGB)
    
    # 2. Inference
    with st.spinner("Analyzing Neural Structure..."):
        results = model.predict(
            image_rgb, 
            conf=conf_threshold, 
            iou=iou_threshold,
            retina_masks=True
        )
        result = results[0]

    # 3. Process Results
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Scan")
        st.image(image_rgb, use_container_width=True, channels="RGB")

    # Generate Overlay
    # We draw custom masks to handle opacity better than default plot()
    annotated_img = image_rgb.copy()
    
    detections_found = False
    
    tumor_data = []

    if result.masks:
        detections_found = True
        
        # Get polygons
        for i, mask in enumerate(result.masks.xy):
            cls_id = int(result.boxes.cls[i])
            conf_score = float(result.boxes.conf[i])
            label_name = result.names[cls_id]
            
            # Draw filled polygon (Mask)
            pts = np.array(mask, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Create a separate overlay for transparency
            overlay = annotated_img.copy()
            color = (255, 50, 50) # Red for tumor
            
            cv2.fillPoly(overlay, [pts], color)
            cv2.addWeighted(overlay, mask_opacity, annotated_img, 1 - mask_opacity, 0, annotated_img)
            
            # Draw Border
            cv2.polylines(annotated_img, [pts], True, color, line_width)
            
            # Collect Stats
            # Area approximation: Polygon area using Shoelace formula inside cv2
            area_px = cv2.contourArea(pts)
            tumor_data.append({
                "ID": i+1,
                "Type": label_name,
                "Confidence": f"{conf_score:.2f}",
                "Area (px)": f"{area_px:.0f}"
            })
            
    elif result.boxes:
        # Fallback for detection-only models (no segmentation)
        detections_found = True
        annotated_img = result.plot() # Use default plotter
        st.warning("⚠️ Model returned boxes only (Detection), not Segmentation masks.")
    
    with col2:
        st.subheader("AI Analysis Output")
        st.image(annotated_img, use_container_width=True, channels="RGB")

    # 4. Metrics & Reporting
    st.markdown("---")
    st.subheader("📊 Analysis Report")
    
    if detections_found and tumor_data:
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Tumors Detected", len(tumor_data))
        m_col2.metric("Max Confidence", max([d['Confidence'] for d in tumor_data]))
        m_col3.metric("Total Area (pixels)", sum([float(d['Area (px)']) for d in tumor_data]))
        
        st.table(pd.DataFrame(tumor_data))
        
        # Download Button
        res_pil = PIL.Image.fromarray(annotated_img)
        buf = io.BytesIO()
        res_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()
        
        st.download_button(
            label="⬇️ Download Segmented Image", 
            data=byte_im,
            file_name="neuroscan_result.png",
            mime="image/png",
            use_container_width=True
        )
    elif detections_found:
        st.info("Tumor detected, but no geometric data allowed for advanced metrics.")
    else:
        st.success("✅ No tumors detected at current confidence levels.")

else:
    # Empty State Hero
    st.markdown("""
    <div style="display: flex; flex-direction: column; align-items: center; justify-content: center; height: 300px; background-color: rgba(255,255,255,0.05); border-radius: 15px;">
        <h2 style="color: #64748b;">Waiting for Scan...</h2>
        <p style="color: #94a3b8;">Upload a file to begin the NeuroScan analysis</p>
    </div>
    """, unsafe_allow_html=True)
