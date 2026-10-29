
from pathlib import Path
import pickle

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

# ---------------------------------
# 1. Project configuration
# ---------------------------------
st.set_page_config(
    page_title="Vehicle Type Classification",
    page_icon="🚘",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "vehicle_ann_model.keras"
ENCODER_PATH = BASE_DIR / "vehicle_label_encoder.pkl"

IMG_SIZE = (64, 64)

# ---------------------------------
# 2. Application heading
# ---------------------------------
st.title("🚘 Vehicle Type Classification")
st.write(
    "Upload a vehicle image to predict its category "
    "using an Artificial Neural Network (ANN)."
)

# ---------------------------------
# 3. Check model files
# ---------------------------------
if not MODEL_PATH.is_file():
    st.error(f"Model file not found: {MODEL_PATH}")
    st.info(
        "Place vehicle_ann_model.keras in the same "
        "folder as app.py."
    )
    st.stop()

if not ENCODER_PATH.is_file():
    st.error(f"Label encoder file not found: {ENCODER_PATH}")
    st.stop()

# ---------------------------------
# 4. Load model and label encoder
# ---------------------------------
@st.cache_resource
def load_model_and_encoder():
    model = tf.keras.models.load_model(
        str(MODEL_PATH),
        compile=False,
        safe_mode=True
    )

    with open(ENCODER_PATH, "rb") as file:
        label_encoder = pickle.load(file)

    return model, label_encoder


try:
    model, label_encoder = load_model_and_encoder()
    st.success("Model and label encoder loaded successfully!")

except Exception as error:
    st.error("Could not load the model files.")
    st.exception(error)
    st.info(
        "Check the Keras/TensorFlow versions used during "
        "training and deployment. If needed, re-save the "
        "model in a compatible environment."
    )
    st.stop()

# ---------------------------------
# 5. Upload image
# ---------------------------------
uploaded_file = st.file_uploader(
    "Choose a vehicle image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:

    try:
        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image).convert("RGB")

        st.subheader("Uploaded Image")
        st.image(image, caption="Input vehicle image",
                 width=350)

        # ---------------------------------
        # 6. Image preprocessing
        # Must match training preprocessing
        # ---------------------------------
        processed_image = image.resize(
            IMG_SIZE,
            Image.Resampling.LANCZOS
        )

        image_array = np.asarray(
            processed_image,
            dtype=np.float32
        )

        # Normalize pixel values from 0-255 to 0-1
        image_array = image_array / 255.0

        # Add batch dimension
        image_array = np.expand_dims(image_array, axis=0)

        # ---------------------------------
        # 7. Predict vehicle category
        # ---------------------------------
        if st.button("🔍 Predict Vehicle Type", type="primary"):

            with st.spinner("Predicting vehicle category..."):
                predictions = model.predict(
                    image_array,
                    verbose=0
                )

            probabilities = np.asarray(predictions)[0]

            if probabilities.ndim != 1:
                st.error("Unexpected model output shape.")
                st.stop()

            predicted_index = int(np.argmax(probabilities))

            # Verify output and encoder class counts
            if len(probabilities) != len(label_encoder.classes_):
                st.error(
                    "Model output classes do not match the "
                    "label encoder classes. Use the matching "
                    "encoder saved during training."
                )
                st.stop()

            predicted_label = label_encoder.inverse_transform(
                [predicted_index]
            )[0]

            confidence = float(
                probabilities[predicted_index] * 100
            )

            # ---------------------------------
            # 8. Display prediction
            # ---------------------------------
            st.subheader("Prediction Result")

            st.metric(
                label="Predicted Vehicle",
                value=str(predicted_label)
            )

            st.metric(
                label="Model Confidence",
                value=f"{confidence:.2f}%"
            )

            if confidence < 60:
                st.warning(
                    "The model is not very confident. "
                    "Please interpret this prediction cautiously."
                )

            # ---------------------------------
            # 9. Show class probabilities
            # ---------------------------------
            st.subheader("Class-wise Prediction Scores")

            class_names = label_encoder.classes_

            for index, class_name in enumerate(class_names):
                score = float(probabilities[index] * 100)

                st.write(f"**{class_name}** — {score:.2f}%")
                st.progress(
                    min(max(score / 100.0, 0.0), 1.0)
                )

    except Exception as error:
        st.error("Unable to process this image.")
        st.exception(error)

st.divider()
st.caption(
    "Project: Vehicle Type Classification Using ANN | "
    "Predictions depend on the model's training and accuracy."
)
