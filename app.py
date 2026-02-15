import streamlit as st
import numpy as np
from PIL import Image

from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout


# ===============================
# CONFIG
# ===============================

IMAGE_SIZE = 128
MODEL_PATH = "models/model.keras"   # uses your uploaded model

# IMPORTANT: Alphabetical order used by flow_from_directory
class_labels = [
    "Glioma Tumor",
    "No Tumor",
    "Pituitary Tumor",
    "Meningioma Tumor"
]



# ===============================
# LOAD MODEL (ARCH + WEIGHTS)
# ===============================

@st.cache_resource
def load_model_correct():
    base_model = VGG16(
        weights=None,
        include_top=False,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)
    )

    model = Sequential([
        base_model,
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(4, activation="softmax")
    ])

    model.load_weights(MODEL_PATH)
    return model


model = load_model_correct()


# ===============================
# PREDICTION FUNCTION
# ===============================

def predict(image: Image.Image):
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    img_array = np.array(image, dtype=np.float32)
    img_array = img_array / 255.0   # EXACT training preprocessing
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)

    index = np.argmax(predictions)
    confidence = np.max(predictions)

    return class_labels[index], confidence, predictions[0]


# ===============================
# STREAMLIT UI
# ===============================

st.set_page_config(
    page_title="MRI Brain Tumor Detection",
    page_icon="🧠",
    layout="centered"
)

st.title("🧠 MRI Brain Tumor Detection")
st.write("Upload an MRI scan to detect the presence and type of brain tumor.")

uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded MRI Image",
        use_container_width=True
    )

    with st.spinner("Analyzing MRI scan..."):
        result, confidence, probs = predict(image)

    st.success(f"Prediction: **{result}**")
    st.info(f"Confidence: **{confidence * 100:.2f}%**")

    # Optional probability breakdown
    st.subheader("Class Probabilities")
    for label, prob in zip(class_labels, probs):
        st.write(f"{label}: {prob * 100:.2f}%")
