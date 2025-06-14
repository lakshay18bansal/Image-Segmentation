import os
import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from torchvision import transforms
import matplotlib.cm as cm
import cv2

# ----------------- CONFIG -----------------
st.set_page_config(page_title="Self-Driving Car Segmentation", layout="centered")
st.markdown("<h1 style='text-align: center;'>🚘 Road Scene Segmentation</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a car camera image to view Segmentation output.</p>", unsafe_allow_html=True)

# ----------------- DOWNLOAD MODEL FROM DRIVE -----------------
MODEL_PATH = "deeplabv3_epoch_100.h5"
DRIVE_FILE_ID = "1kaNa4emUrrYIDKQkXIdFU_FAgpZr8Sed"

def download_model():
    import gdown
    url = f"https://drive.google.com/uc?id={DRIVE_FILE_ID}"
    with st.spinner("📥 Downloading model file from Google Drive..."):
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

# ----------------- SEGMENTATION FUNCTION -----------------
def perform_image_seg(model, image):
    input_size = 256
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image)
    image_np = image_tensor.permute(1, 2, 0).numpy()
    image_tf = tf.convert_to_tensor(np.expand_dims(image_np, axis=0), dtype=tf.float32)
    
    pred_logits = model.predict(image_tf)
    pred_mask = tf.argmax(pred_logits, axis=-1).numpy()[0]
    
    return pred_mask

# ----------------- IMAGE UPLOAD -----------------
uploaded_file = st.file_uploader("Upload a driving image", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    with st.spinner("Segmenting image..."):
        mask = perform_image_seg(model, image)

    colored_mask = cm.get_cmap('jet')(mask / (mask.max() if mask.max() > 0 else 1))[:, :, :3]
    colored_mask = (colored_mask * 255).astype(np.uint8)

    image_resized = image.resize((256, 256))
    image_np = np.array(image_resized)
    overlay = cv2.addWeighted(image_np, 0.4, colored_mask, 0.6, 10)

    st.markdown("### 🎯 Segmentation Output")
    st.image(overlay, caption="Overlayed Prediction", use_container_width=True)

    st.download_button(
        label="Download Result",
        data=cv2.imencode(".png", overlay)[1].tobytes(),
        file_name="segmented_output.png",
        mime="image/png"
    )
else:
    st.info("👈 Upload a car image (e.g., from dashcam or dataset) to get started.")
