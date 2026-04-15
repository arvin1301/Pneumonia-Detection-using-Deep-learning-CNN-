import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import cv2
import tempfile

# -----------------------------
# FIX FOR quantization_config ERROR
# -----------------------------
from tensorflow.keras.layers import Dense

class CustomDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)  # 🔥 remove problematic arg
        super().__init__(*args, **kwargs)

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 180

# -----------------------------
# LOAD MODEL (FIXED)
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "best_model.h5",
        custom_objects={"Dense": CustomDense},  # 🔥 FIX
        compile=False
    )

model = load_model()

# -----------------------------
# LOAD CLASS NAMES
# -----------------------------
with open("class_names.json", "r") as f:
    class_names = json.load(f)

# -----------------------------
# PREDICTION FUNCTION
# -----------------------------
def predict_image(image):
    img = image.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img) / 255.0

    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    pred_class = class_names[np.argmax(prediction)]
    confidence = float(np.max(prediction))

    return pred_class, confidence

# -----------------------------
# UI
# -----------------------------
st.title("🩺 Pneumonia Detection")

option = st.sidebar.selectbox(
    "Select Input",
    ["Image Upload", "Webcam", "Video"]
)

# IMAGE
if option == "Image Upload":
    file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])
    if file:
        image = Image.open(file).convert("RGB")
        st.image(image)

        pred, conf = predict_image(image)
        st.success(f"{pred}")
        st.info(f"Confidence: {conf:.2f}")

# WEBCAM
elif option == "Webcam":
    img_file = st.camera_input("Capture")
    if img_file:
        image = Image.open(img_file)
        st.image(image)

        pred, conf = predict_image(image)
        st.success(f"{pred}")
        st.info(f"Confidence: {conf:.2f}")

# VIDEO
elif option == "Video":
    video = st.file_uploader("Upload Video", type=["mp4","avi"])
    if video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(rgb)

            pred, conf = predict_image(img)

            cv2.putText(frame, f"{pred} ({conf:.2f})",
                        (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0,255,0), 2)

            stframe.image(frame, channels="BGR")

        cap.release()