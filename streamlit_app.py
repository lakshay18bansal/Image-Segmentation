import os
import streamlit as st
from PIL import Image
import numpy as np
import onnxruntime as ort
import cv2
import matplotlib.cm as cm

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Self-Driving Car Segmentation", layout="centered")
st.markdown("<h1 style='text-align: center;'>🚘 Road Scene Segmentation</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a car camera image to view segmentation output.</p>", unsafe_allow_html=True)

# ----------------- DOWNLOAD MODEL FROM GOOGLE DRIVE -----------------
MODEL_PATH = "deeplabv3_model.onnx"
DRIVE_FILE_ID = "1MRBng1vQffR4Z2hoFjcmFcMNQWeFozdg"

def download_model():
    import gdown
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    st.info("📥Downloading files...")
    gdown.download(url, MODEL_PATH, quiet=False)

if not os.path.exists(MODEL_PATH):
    try:
        import gdown
    except ImportError:
        os.system("pip install gdown")
        import gdown
    download_model()

# ----------------- LOAD MODEL -----------------
@st.cache_resource(show_spinner="Loading...")
def load_model():
    return ort.InferenceSession(MODEL_PATH)

model = load_model()

# ----------------- SEGMENTATION -----------------
def perform_image_seg(model, pil_image, input_size=256):
    image = pil_image.convert('RGB')
    image_resized = image.resize((input_size, input_size), Image.BILINEAR)
    
    image_np = np.array(image_resized).astype(np.float32) / 255.0  # (256, 256, 3)
    input_tensor = np.expand_dims(image_np, axis=0)  # (1, 256, 256, 3)

    input_name = model.get_inputs()[0].name
    
    # Optional: log shape in Streamlit
    st.write("📐 Input shape to model:", input_tensor.shape)

    # Inference
    output = model.run(None, {input_name: input_tensor})[0]  # (1, 256, 256, 35)
    pred_mask = np.argmax(output, axis=-1)[0]  # (256, 256)
    
    return pred_mask, image_resized

# ----------------- OVERLAY FUNCTION -----------------
def overlay_mask_with_edges(original_resized, mask):
    # Colorize mask using jet colormap
    mask_colored = cm.get_cmap('jet')(mask / mask.max())[:, :, :3]
    mask_colored = (mask_colored * 255).astype(np.uint8)

    # Compute edges on original image
    gray = cv2.cvtColor(np.array(original_resized), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_overlay = np.zeros_like(mask_colored)
    edge_overlay[edges > 0] = [255, 0, 0]  # Red

    # Combine mask and edges
    combined = cv2.addWeighted(mask_colored, 0.8, edge_overlay, 1.0, 0)

    # Blend with original image (light context)
    blended = cv2.addWeighted(np.array(original_resized), 0.25, combined, 0.75, 0)

    return blended

# ----------------- STREAMLIT UI -----------------
uploaded_file = st.file_uploader("Upload a driving image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image.thumbnail((512, 512), Image.BILINEAR)
    
    st.image(image, caption="📷 Uploaded Image", width=512)

    with st.spinner("Segmenting image..."):
        mask, resized_img = perform_image_seg(model, image)
        overlay = overlay_mask_with_edges(resized_img, mask)

    st.markdown("### 🎯 Segmentation Output")
    st.image(overlay, caption="Overlayed Prediction with Edges (512×512)", width=512)

    st.download_button(
        label="Download Result",
        data=cv2.imencode(".png", overlay)[1].tobytes(),
        file_name="segmented_output.png",
        mime="image/png"
    )
else:
    st.info("👈 Upload a car image (e.g., from dashcam or dataset) to get started.")
