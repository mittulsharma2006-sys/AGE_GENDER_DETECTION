import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os

from keras.src.legacy.saving import legacy_h5_format

import sys
import os

TASK_1_PATH = os.path.join(
    os.path.dirname(__file__),
    "Task-1-Car-Colour-Detection"
)

sys.path.insert(0, TASK_1_PATH)

from modules.traffic_detection import TrafficDetector


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Detection System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
<style>

.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1E3A8A;
    text-align: center;
    margin-bottom: 2rem;
}

.sub-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: #2563EB;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}

.result-text {
    font-size: 1.5rem;
    font-weight: 500;
    padding: 0.75rem;
    border-radius: 0.5rem;
    margin-bottom: 0.5rem;
}

.image-container {
    margin-bottom: 2rem;
    padding: 1rem;
    border-radius: 0.5rem;
    background-color: rgba(237, 242, 247, 0.5);
}

.app-footer {
    text-align: center;
    margin-top: 2rem;
    opacity: 0.7;
}

.stButton>button {
    background-color: #2563EB;
    color: white;
    font-weight: bold;
    border-radius: 0.5rem;
    padding: 0.5rem 1rem;
    border: none;
}

.stButton>button:hover {
    background-color: #1E40AF;
}

</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# AGE & GENDER MODEL
# =========================================================

@st.cache_resource
def load_age_gender_model():

    try:

        # Model is inside the GitHub project folder
        model_path = os.path.join(
            os.path.dirname(__file__),
            "Age_Sex_Detection.h5"
        )

        model = legacy_h5_format.load_model_from_hdf5(
            model_path,
            custom_objects={"mae": "mae"}
        )

        return model

    except Exception as e:

        st.error(
            f"Error loading Age/Gender model: {e}"
        )

        return None


# =========================================================
# TRAFFIC MODEL
# =========================================================

@st.cache_resource
def load_traffic_model():

    try:

        detector = TrafficDetector()

        return detector

    except Exception as e:

        st.error(
            f"Error loading traffic model: {e}"
        )

        return None


# =========================================================
# AGE & GENDER FUNCTIONS
# =========================================================

def preprocess_image(uploaded_image):

    if uploaded_image.mode != "RGB":

        uploaded_image = uploaded_image.convert("RGB")

    image = uploaded_image.resize(
        (48, 48)
    )

    image_array = np.array(image) / 255.0

    return np.expand_dims(
        image_array,
        axis=0
    )


def predict_age_gender(
    model,
    image_array
):

    try:

        predictions = model.predict(
            image_array,
            verbose=0
        )

        # Age
        predicted_age = int(
            np.round(
                predictions[1][0]
            )
        )

        # Gender
        gender_prob = predictions[0][0]

        predicted_gender = (
            "Female"
            if gender_prob > 0.5
            else "Male"
        )

        gender_confidence = (
            gender_prob
            if predicted_gender == "Female"
            else 1 - gender_prob
        )

        return (
            predicted_age,
            predicted_gender,
            float(gender_confidence)
        )

    except Exception as e:

        st.error(
            f"Error during prediction: {e}"
        )

        return None, None, None


# =========================================================
# AGE & GENDER PAGE
# =========================================================

def age_gender_page():

    st.markdown(
        '<div class="main-header">'
        'Age & Gender Detector'
        '</div>',
        unsafe_allow_html=True
    )

    # Load model only for this page
    with st.spinner(
        "Loading Age & Gender model..."
    ):

        model = load_age_gender_model()

    if model is None:

        st.warning(
            "Age/Gender model could not be loaded."
        )

        return

    st.markdown(
        '<div class="sub-header">'
        'Upload Images'
        '</div>',
        unsafe_allow_html=True
    )

    uploaded_files = st.file_uploader(
        "Choose one or more images",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        accept_multiple_files=True,
        key="age_gender_uploader"
    )

    if uploaded_files:

        if st.button(
            "Detect Age & Gender",
            key="age_gender_button"
        ):

            with st.spinner(
                "Analyzing images..."
            ):

                for i, uploaded_file in enumerate(
                    uploaded_files
                ):

                    with st.container():

                        st.markdown(
                            '<div class="image-container">',
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            f"<h3>Image {i + 1}</h3>",
                            unsafe_allow_html=True
                        )

                        col1, col2 = st.columns(
                            [1, 1]
                        )

                        image = Image.open(
                            uploaded_file
                        )

                        # Input image
                        col1.image(
                            image,
                            caption=(
                                f"Input: "
                                f"{uploaded_file.name}"
                            ),
                            use_column_width=True
                        )

                        # Process
                        processed_image = (
                            preprocess_image(image)
                        )

                        age, gender, confidence = (
                            predict_age_gender(
                                model,
                                processed_image
                            )
                        )

                        if (
                            age is not None
                            and gender is not None
                        ):

                            col2.markdown(
                                '<div class="sub-header">'
                                'Results'
                                '</div>',
                                unsafe_allow_html=True
                            )

                            col2.metric(
                                "Predicted Age",
                                f"{age} years"
                            )

                            col2.metric(
                                "Gender",
                                gender
                            )

                            col2.metric(
                                "Confidence",
                                f"{confidence:.2%}"
                            )

                        else:

                            col2.error(
                                "Failed to process image."
                            )

                        st.markdown(
                            '</div>',
                            unsafe_allow_html=True
                        )


# =========================================================
# TRAFFIC ANALYSIS PAGE
# =========================================================

def traffic_page():

    st.markdown(
        '<div class="main-header">'
        '🚦 Traffic Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.write(
        """
        Upload a traffic image to detect cars and people,
        count vehicles and people, and identify car colours.
        """
    )

    # Load traffic detector
    with st.spinner(
        "Loading traffic detection model..."
    ):

        detector = load_traffic_model()

    if detector is None:

        st.error(
            "Traffic detection model could not be loaded."
        )

        return

    # Upload image
    uploaded_file = st.file_uploader(
        "Upload a traffic image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="traffic_uploader"
    )

    if uploaded_file is None:

        st.info(
            "Please upload a traffic image to begin."
        )

        return

    # Open image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # -----------------------------------------------------
    # INPUT IMAGE
    # -----------------------------------------------------

    st.markdown(
        '<div class="sub-header">'
        'Input Image'
        '</div>',
        unsafe_allow_html=True
    )

    st.image(
        image,
        caption="Uploaded Traffic Image",
        use_column_width=True
    )

    # -----------------------------------------------------
    # ANALYZE BUTTON
    # -----------------------------------------------------

    if st.button(
        "🚗 Analyze Traffic",
        key="traffic_button"
    ):

        with st.spinner(
            "Detecting cars, colours and people..."
        ):

            image_array = np.array(
                image
            )

            result, statistics = (
                detector.process_image(
                    image_array
                )
            )

        st.success(
            "Traffic analysis completed!"
        )

        # -------------------------------------------------
        # RESULT IMAGE
        # -------------------------------------------------

        st.markdown(
            '<div class="sub-header">'
            'Detection Result'
            '</div>',
            unsafe_allow_html=True
        )

        st.image(
            result,
            caption="Traffic Detection Result",
            use_column_width=True
        )

        # -------------------------------------------------
        # STATISTICS
        # -------------------------------------------------

        st.markdown(
            '<div class="sub-header">'
            'Detection Statistics'
            '</div>',
            unsafe_allow_html=True
        )

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "🚗 Total Cars",
            statistics["cars"]
        )

        col2.metric(
            "🔴 Blue Cars",
            statistics["blue_cars"]
        )

        col3.metric(
            "🔵 Other Cars",
            statistics["other_cars"]
        )

        col4.metric(
            "👤 People",
            statistics["people"]
        )

        # -------------------------------------------------
        # EXPLANATION
        # -------------------------------------------------

        st.markdown(
            """
            ### Bounding Box Rules

            🔴 **Red rectangle** → Blue car

            🔵 **Blue rectangle** → Other colour car

            🟢 **Green rectangle** → Person
            """
        )


# =========================================================
# MAIN APPLICATION
# =========================================================

def main():

    # -----------------------------------------------------
    # SIDEBAR
    # -----------------------------------------------------

    st.sidebar.title(
        "🤖 AI Detection System"
    )

    st.sidebar.markdown(
        "---"
    )

    selected_mode = st.sidebar.radio(
        "Select Application",
        [
            "Age & Gender Detection",
            "Traffic Analysis"
        ]
    )

    st.sidebar.markdown(
        "---"
    )

    st.sidebar.info(
        """
        This project combines the
        original Age & Gender Detection
        system with the internship
        Traffic Analysis extension.
        """
    )

    # -----------------------------------------------------
    # SELECT PAGE
    # -----------------------------------------------------

    if selected_mode == "Age & Gender Detection":

        age_gender_page()

    elif selected_mode == "Traffic Analysis":

        traffic_page()

    # -----------------------------------------------------
    # FOOTER
    # -----------------------------------------------------

    st.markdown(
        '<div class="app-footer">'
        'Powered by NULLCLASS 🧑‍💻'
        '</div>',
        unsafe_allow_html=True
    )


# =========================================================
# RUN APPLICATION
# =========================================================

if __name__ == "__main__":

    main()