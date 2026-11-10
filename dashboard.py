
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
from PIL import Image

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="Diabetic Retinopathy Dashboard",
    page_icon="👁️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
MODEL_DIR = BASE_DIR / "models"

CLASS_NAMES = [
    "Mild",
    "Moderate",
    "No DR",
    "Proliferate DR",
    "Severe"
]

# -------------------------------
# PAGE TITLE
# -------------------------------
st.title("👁️ Diabetic Retinopathy Classification")
st.subheader("Results and Evaluation Dashboard")
st.caption("CNN | TensorFlow | Five-Class Classification")

if not OUTPUT_DIR.exists():
    st.error("Output folder not found.")
    st.stop()

# -------------------------------
# HELPER FUNCTIONS
# -------------------------------
def show_image(filename, title):
    path = OUTPUT_DIR / filename

    st.subheader(title)

    if path.exists():
        st.image(str(path), use_column_width=True)
    else:
        st.info(f"{filename} not found.")


def show_sample_predictions():
    folder = OUTPUT_DIR / "sample_predictions"

    if not folder.exists():
        st.info("Sample predictions folder not found.")
        return

    files = sorted(folder.glob("*.png"))

    if not files:
        st.info("No prediction images found.")
        return

    cols = st.columns(2)

    for i, file in enumerate(files):
        with cols[i % 2]:
            st.image(
                str(file),
                caption=file.name,
                use_column_width=True
            )


@st.cache_resource
def load_trained_model(model_path):
    return tf.keras.models.load_model(str(model_path))


# -------------------------------
# DASHBOARD TABS
# -------------------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Training Graphs",
    "Model Evaluation",
    "Predictions",
    "Error Analysis"
])

# -------------------------------
# TAB 1: OVERVIEW
# -------------------------------
with tab1:
    st.header("Project Overview")

    st.write(
        "Deep Learning-Based Diabetic Retinopathy Classification"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Total Images", "35,126")
    c2.metric("Training Images", "28,100")
    c3.metric("Testing Images", "7,026")

    st.subheader("Classification Categories")

    class_counts = {
        "Mild": 2443,
        "Moderate": 5292,
        "No DR": 25810,
        "Proliferate DR": 708,
        "Severe": 873
    }

    st.bar_chart(
        pd.Series(class_counts, name="Number of Images")
    )

    st.subheader("Dataset Information")

    st.write("Number of classes: 5")
    st.write("Image size: 128 × 128")
    st.write("Training split: 80%")
    st.write("Testing split: 20%")

# -------------------------------
# TAB 2: TRAINING GRAPHS
# -------------------------------
with tab2:
    st.header("Training Performance")

    show_image(
        "accuracy_graph.png",
        "Training vs Validation Accuracy"
    )

    show_image(
        "loss_graph.png",
        "Training vs Validation Loss"
    )

# -------------------------------
# TAB 3: MODEL EVALUATION
# -------------------------------
with tab3:
    st.header("Model Evaluation")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", "56.16%")
    c2.metric("Precision", "23.00%")
    c3.metric("Recall", "23.53%")
    c4.metric("F1 Score", "22.80%")

    show_image(
        "confusion_matrix.png",
        "Confusion Matrix"
    )

    show_image(
        "metrics_comparison.png",
        "Evaluation Metrics Comparison"
    )

    report_path = OUTPUT_DIR / "classification_report.txt"

    if report_path.exists():
        st.subheader("Class-wise Classification Report")

        st.text(
            report_path.read_text(
                encoding="utf-8",
                errors="replace"
            )
        )
    else:
        st.info("Classification report not found.")

# -------------------------------
# TAB 4: PREDICTIONS
# -------------------------------
with tab4:
    st.header("Sample Prediction Results")

    show_sample_predictions()

    st.divider()

    st.subheader("Upload a Retinal Image")

    uploaded_file = st.file_uploader(
        "Choose a retinal image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        try:
            uploaded_image = Image.open(
                uploaded_file
            ).convert("RGB")

            st.image(
                uploaded_image,
                caption="Uploaded Retinal Image",
                width=350
            )

            if st.button("Predict Diabetic Retinopathy"):

                model_path = (
                    MODEL_DIR / "diabetic_retinopathy_model.keras"
                )

                if not model_path.exists():
                    st.error("Trained model not found.")

                else:
                    with st.spinner("Analyzing image..."):

                        model = load_trained_model(model_path)

                        resized_image = uploaded_image.resize(
                            (128, 128)
                        )

                        image_array = np.array(
                            resized_image,
                            dtype=np.float32
                        )

                        image_array = np.expand_dims(
                            image_array,
                            axis=0
                        )

                        prediction = model.predict(
                            image_array,
                            verbose=0
                        )

                        predicted_index = int(
                            np.argmax(prediction[0])
                        )

                        predicted_class = CLASS_NAMES[
                            predicted_index
                        ]

                        confidence = float(
                            np.max(prediction[0]) * 100
                        )

                    st.success(
                        f"Predicted Class: {predicted_class}"
                    )

                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

                    st.caption(
                        "For educational and research purposes only. "
                        "This prediction is not a medical diagnosis."
                    )

        except Exception as e:
            st.error(f"Error processing image: {e}")

# -------------------------------
# TAB 5: ERROR ANALYSIS
# -------------------------------
with tab5:
    st.header("Error Analysis")

    st.write(
        "Review the model's performance across the five classes."
    )

    show_image(
        "confusion_matrix.png",
        "Classification Errors"
    )

    st.subheader("Sample Predictions")

    show_sample_predictions()

    st.info(
        "The model currently performs better on the No DR class. "
        "Minority classes need further improvement."
    )

# -------------------------------
# FOOTER
# -------------------------------
st.divider()

st.caption(
    "Diabetic Retinopathy Classification | "
    "CNN and TensorFlow | Academic Project"
)
