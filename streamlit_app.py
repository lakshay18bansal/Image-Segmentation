import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.cm as cm

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Self-Driving Car Segmentation", layout="centered")
st.markdown("<h1 style='text-align: center;'>🚘 Road Scene Segmentation</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a car camera image to view segmentation output.</p>", unsafe_allow_html=True)

# ----------------- DOWNLOAD MODEL FROM GOOGLE DRIVE -----------------
MODEL_PATH = "deeplabv3_epoch_100.h5"
DRIVE_FILE_ID = "1kaNa4emUrrYIDKQkXIdFU_FAgpZr8Sed"

def download_model():
    import gdown
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    st.info("📥 Downloading model file from Google Drive...")
    gdown.download(url, MODEL_PATH, quiet=False)

if not os.path.exists(MODEL_PATH):
    try:
        import gdown
    except ImportError:
        os.system("pip install gdown")
        import gdown
    download_model()

# ----------------- LOAD MODEL -----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

model = load_model()

# ----------------- PREPROCESS INPUT -----------------
def preprocess_image(image, target_size=(256, 256)):
    original_size = image.size  # (width, height)
    image_resized = image.resize(target_size, Image.BILINEAR)
    image_np = np.array(image_resized).astype(np.float32) / 255.0
    return image_np, original_size

# ----------------- POSTPROCESS MASK -----------------
def postprocess_mask(mask, original_size):
    mask_uint8 = (mask * 255).astype(np.uint8)
    mask_resized = cv2.resize(mask_uint8, original_size, interpolation=cv2.INTER_NEAREST)
    return mask_resized

# ----------------- OVERLAY MASK -----------------
def overlay_mask_on_image(image, mask):
    image_np = np.array(image)
    mask_colored = cm.get_cmap('jet')(mask / 255.0)[:, :, :3]
    mask_colored = (mask_colored * 255).astype(np.uint8)
    overlay = cv2.addWeighted(image_np, 0.2, mask_colored, 0.8, 10)
    return overlay

# ----------------- SEGMENTATION FUNCTION -----------------
def perform_segmentation(model, image):
    preprocessed, original_size = preprocess_image(image)
    image_tf = tf.convert_to_tensor(np.expand_dims(preprocessed, axis=0), dtype=tf.float32)
    pred_logits = model.predict(image_tf)
    pred_mask = tf.argmax(pred_logits, axis=-1).numpy()[0]
    resized_mask = postprocess_mask(pred_mask, original_size)
    return resized_mask

# ----------------- STREAMLIT UI -----------------
uploaded_file = st.file_uploader("Upload a driving image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 Uploaded Image", width=512)

    with st.spinner("Segmenting image..."):
        mask = perform_segmentation(model, image)

    overlay = overlay_mask_on_image(image, mask)

    st.markdown("### 🎯 Segmentation Output")
    st.image(overlay, caption="Overlayed Prediction", width=512)

    st.download_button(
        label="Download Result",
        data=cv2.imencode(".png", overlay)[1].tobytes(),
        file_name="segmented_output.png",
        mime="image/png"
    )
else:
    st.info("👈 Upload a car image (e.g., from dashcam or dataset) to get started.")
