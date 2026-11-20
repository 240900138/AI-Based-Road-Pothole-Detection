
import streamlit as st
import pandas as pd
import os

from PIL import Image
from datetime import datetime

from predict import predict_pothole
from config import MODEL_PATH, OUTPUT_DIR


# Page configuration
st.set_page_config(
    page_title="AI Road Pothole Detection",
    page_icon="🛣️",
    layout="wide"
)


# Session history
if "history" not in st.session_state:
    st.session_state.history = []


# Custom styling
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    font-size: 32px;
    font-weight: bold;
    color: #183153;
}

.subtitle {
    color: #64748b;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)


# Sidebar
st.sidebar.title("🛣️ Pothole Detection")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Detect Potholes",
        "Detection History",
        "Model Performance",
        "About System"
    ]
)


# Dashboard
if page == "Dashboard":

    st.markdown(
        '<p class="title">AI-Based Road Pothole Detection System</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Machine Learning powered road condition analysis</p>',
        unsafe_allow_html=True
    )

    total = len(st.session_state.history)

    potholes = sum(
        1 for item in st.session_state.history
        if item["Prediction"] == "Pothole Detected"
    )

    normal = total - potholes

    col1, col2, col3 = st.columns(3)

    col1.metric("Images Analysed", total)
    col2.metric("Potholes Detected", potholes)
    col3.metric("Normal Roads", normal)

    st.divider()

    st.subheader("System Overview")

    st.write("""
    The AI-Based Road Pothole Detection System uses
    Machine Learning and Computer Vision to analyse
    road images and identify possible potholes.

    The system assists in identifying road damage
    that may require municipal inspection and repair.
    """)

    st.info(
        "Navigate to Detect Potholes to upload and analyse a road image."
    )


# Detection page
elif page == "Detect Potholes":

    st.title("Road Pothole Detection")

    st.write(
        "Upload a road image to analyse its condition."
    )

    if not os.path.exists(MODEL_PATH):

        st.error(
            "Trained model not found. "
            "Please run train.py first."
        )

    else:

        uploaded_file = st.file_uploader(
            "Upload Road Image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:

            image = Image.open(uploaded_file)

            st.image(
                image,
                caption="Uploaded Road Image",
                width=500
            )

            if st.button(
                "Analyse Road Image",
                type="primary"
            ):

                with st.spinner(
                    "Analysing road image..."
                ):

                    result = predict_pothole(image)

                st.divider()

                st.subheader("Detection Results")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Prediction",
                    result["prediction"]
                )

                col2.metric(
                    "Confidence",
                    f'{result["confidence"]}%'
                )

                if result["prediction"] == "Pothole Detected":

                    st.error(result["recommendation"])

                else:

                    st.success(result["recommendation"])

                # Save history
                st.session_state.history.append({

                    "Date": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                    "Image": uploaded_file.name,

                    "Prediction": result["prediction"],

                    "Confidence": result["confidence"]

                })


# Detection history
elif page == "Detection History":

    st.title("Detection History")

    if len(st.session_state.history) > 0:

        df = pd.DataFrame(
            st.session_state.history
        )

        st.dataframe(
            df,
            use_container_width=True
        )

        csv = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download History as CSV",
            csv,
            "pothole_detection_history.csv",
            "text/csv"
        )

    else:

        st.info(
            "No detection records available yet."
        )


# Model performance
elif page == "Model Performance":

    st.title("Machine Learning Model Performance")

    report_path = os.path.join(
        OUTPUT_DIR,
        "classification_report.txt"
    )

    if os.path.exists(report_path):

        with open(report_path, "r") as file:

            st.text(file.read())

    else:

        st.warning(
            "Model evaluation results are not available. "
            "Run evaluate.py first."
        )

    st.subheader("Training Accuracy")

    accuracy_path = os.path.join(
        OUTPUT_DIR,
        "training_accuracy.png"
    )

    if os.path.exists(accuracy_path):

        st.image(accuracy_path)

    st.subheader("Training Loss")

    loss_path = os.path.join(
        OUTPUT_DIR,
        "training_loss.png"
    )

    if os.path.exists(loss_path):

        st.image(loss_path)

    st.subheader("Confusion Matrix")

    cm_path = os.path.join(
        OUTPUT_DIR,
        "confusion_matrix.png"
    )

    if os.path.exists(cm_path):

        st.image(cm_path)


# About system
elif page == "About System":

    st.title("About the System")

    st.write("""
    The AI-Based Road Pothole Detection System is a
    Machine Learning application designed to identify
    potholes from road images.

    Technologies used:

    - Python
    - TensorFlow and Keras
    - Convolutional Neural Networks (CNN)
    - Computer Vision
    - Streamlit
    - Pandas
    - Scikit-learn

    The system classifies uploaded road images into
    two categories: Pothole and Normal Road.
    """)

    st.warning(
        "Predictions are intended to assist road inspections "
        "and should be verified by municipal personnel."
    )
