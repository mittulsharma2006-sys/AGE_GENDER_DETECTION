import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os

# =========================================================
# OPTIONAL AI LIBRARIES
# =========================================================

try:
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except Exception:
    INSIGHTFACE_AVAILABLE = False

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except Exception:
    DEEPFACE_AVAILABLE = False


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Nationality Detection",
    page_icon="🌍",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 17px;
    color: #555;
    margin-bottom: 30px;
}

.section-title {
    font-size: 25px;
    font-weight: 650;
    margin-top: 25px;
    margin-bottom: 15px;
}

.result-box {
    padding: 18px;
    border-radius: 10px;
    margin-bottom: 15px;
    background-color: #eefbf2;
    border: 1px solid #d5f0dc;
}

.result-label {
    font-size: 15px;
    color: #555;
}

.result-value {
    font-size: 23px;
    font-weight: 600;
    margin-top: 5px;
}

.warning-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #fff8e1;
    border: 1px solid #ffe082;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# INSIGHTFACE MODEL
# =========================================================

@st.cache_resource
def load_insightface():

    if not INSIGHTFACE_AVAILABLE:
        return None

    try:

        model = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"]
        )

        model.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

        return model

    except Exception as e:

        st.error(f"Could not load InsightFace: {e}")
        return None


# =========================================================
# FACE / AGE DETECTION
# =========================================================

def predict_age(image):

    model = load_insightface()

    if model is None:
        return None

    try:

        img = np.array(image.convert("RGB"))

        # InsightFace works with BGR images
        img_bgr = cv2.cvtColor(
            img,
            cv2.COLOR_RGB2BGR
        )

        faces = model.get(img_bgr)

        if len(faces) == 0:
            return None

        # Select largest detected face
        face = max(
            faces,
            key=lambda f:
            (f.bbox[2] - f.bbox[0]) *
            (f.bbox[3] - f.bbox[1])
        )

        age = int(round(float(face.age)))

        return age

    except Exception as e:

        st.warning(f"Age detection error: {e}")
        return None


# =========================================================
# EMOTION DETECTION - DEEPFACE
# =========================================================

def predict_emotion(image):

    if not DEEPFACE_AVAILABLE:
        return None

    try:

        img = np.array(image.convert("RGB"))

        result = DeepFace.analyze(
            img_path=img,
            actions=["emotion"],
            detector_backend="opencv",
            enforce_detection=False,
            silent=True
        )

        # DeepFace may return a list
        if isinstance(result, list):
            result = result[0]

        emotion = result.get(
            "dominant_emotion",
            None
        )

        if emotion:
            return emotion.capitalize()

        return None

    except Exception as e:

        st.warning(f"Emotion detection error: {e}")
        return None


# =========================================================
# DRESS COLOUR DETECTION
# =========================================================

def get_dress_colour(image):

    try:

        img = np.array(
            image.convert("RGB")
        )

        height, width = img.shape[:2]

        # Focus on the central/lower portion where clothing
        # is normally visible.
        y1 = int(height * 0.40)
        y2 = int(height * 0.90)

        x1 = int(width * 0.20)
        x2 = int(width * 0.80)

        clothing = img[
            y1:y2,
            x1:x2
        ]

        if clothing.size == 0:
            return "Unknown"

        # Resize for faster processing
        clothing = cv2.resize(
            clothing,
            (200, 200)
        )

        # Convert RGB -> HSV
        hsv = cv2.cvtColor(
            clothing,
            cv2.COLOR_RGB2HSV
        )

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        # Remove very bright background pixels
        valid = (
            (v < 245) &
            (v > 25)
        )

        if np.sum(valid) < 100:
            return "Unknown"

        h_valid = h[valid]
        s_valid = s[valid]
        v_valid = v[valid]

        avg_h = np.mean(h_valid)
        avg_s = np.mean(s_valid)
        avg_v = np.mean(v_valid)

        # -------------------------------------------------
        # LOW SATURATION = BLACK / WHITE / GREY
        # -------------------------------------------------

        if avg_s < 35:

            if avg_v < 70:
                return "Black"

            elif avg_v > 190:
                return "White"

            else:
                return "Grey"

        # -------------------------------------------------
        # COLOUR CLASSIFICATION
        # -------------------------------------------------

        if avg_h < 10 or avg_h >= 170:
            return "Red"

        elif 10 <= avg_h < 25:
            return "Orange"

        elif 25 <= avg_h < 35:
            return "Yellow"

        elif 35 <= avg_h < 85:
            return "Green"

        elif 85 <= avg_h < 135:
            return "Blue"

        elif 135 <= avg_h < 170:
            return "Purple"

        return "Unknown"

    except Exception:
        return "Unknown"


# =========================================================
# RESULT BOX
# =========================================================

def result_box(label, value):

    if value is None:
        value = "Not available"

    st.markdown(
        f"""
        <div class="result-box">
            <div class="result-label">{label}</div>
            <div class="result-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# =========================================================
# MAIN APPLICATION
# =========================================================

def main():

    # -----------------------------------------------------
    # HEADER
    # -----------------------------------------------------

    st.markdown(
        '<div class="main-title">'
        '🌍 Nationality Detection & Attribute Analysis'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'Upload an image and provide the required verified '
        'information to display the relevant attributes.'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    # -----------------------------------------------------
    # INPUT SECTION
    # -----------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            '<div class="section-title">📷 Upload Image</div>',
            unsafe_allow_html=True
        )

        uploaded_file = st.file_uploader(
            "Choose an image",
            type=["jpg", "jpeg", "png"]
        )

    with col2:

        st.markdown(
            '<div class="section-title">📋 Input Information</div>',
            unsafe_allow_html=True
        )

        nationality = st.selectbox(
            "Select verified nationality",
            [
                "Indian",
                "United States",
                "African",
                "Other"
            ]
        )

    # -----------------------------------------------------
    # IMAGE
    # -----------------------------------------------------

    if uploaded_file is None:

        st.info(
            "Please upload an image to start the analysis."
        )

        return

    try:

        image = Image.open(
            uploaded_file
        ).convert("RGB")

    except Exception as e:

        st.error(
            f"Could not open image: {e}"
        )

        return

    # -----------------------------------------------------
    # DISPLAY IMAGE
    # -----------------------------------------------------

    st.markdown(
        '<div class="section-title">🖼️ Uploaded Image</div>',
        unsafe_allow_html=True
    )

    st.image(
        image,
        caption="Input Image",
        width=700
    )

    st.divider()

    # -----------------------------------------------------
    # ANALYZE BUTTON
    # -----------------------------------------------------

    if st.button(
        "🔍 Predict Attributes",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing image..."
        ):

            # =============================================
            # EMOTION
            # =============================================

            emotion = predict_emotion(image)

            # =============================================
            # AGE
            # =============================================

            age = None

            if nationality in [
                "Indian",
                "United States"
            ]:

                age = predict_age(image)

            # =============================================
            # DRESS COLOUR
            # =============================================

            dress_colour = None

            if nationality in [
                "Indian",
                "African"
            ]:

                dress_colour = get_dress_colour(
                    image
                )

        # -------------------------------------------------
        # RESULTS
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '📊 Prediction Results'
            '</div>',
            unsafe_allow_html=True
        )

        # -------------------------------------------------
        # MAIN RESULTS
        # -------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:
            result_box(
                "🌍 Nationality",
                nationality
            )

        with col2:
            result_box(
                "😊 Emotion",
                emotion
            )

        # -------------------------------------------------
        # ADDITIONAL ATTRIBUTES
        # -------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            '🔎 Additional Attributes'
            '</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:

            if nationality in [
                "Indian",
                "United States"
            ]:

                if age is not None:

                    result_box(
                        "🎂 Age",
                        f"{age} years"
                    )

                else:

                    result_box(
                        "🎂 Age",
                        "Face not detected"
                    )

            else:

                result_box(
                    "🎂 Age",
                    "Not required"
                )

        with col2:

            if nationality in [
                "Indian",
                "African"
            ]:

                result_box(
                    "👕 Dress Colour",
                    dress_colour
                )

            else:

                result_box(
                    "👕 Dress Colour",
                    "Not required"
                )

        # -------------------------------------------------
        # SUMMARY
        # -------------------------------------------------

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '📋 Summary'
            '</div>',
            unsafe_allow_html=True
        )

        st.write(
            f"**Nationality:** {nationality}"
        )

        if emotion:
            st.write(
                f"**Emotion:** {emotion}"
            )
        else:
            st.write(
                "**Emotion:** Not detected"
            )

        if nationality in [
            "Indian",
            "United States"
        ]:

            if age is not None:

                st.write(
                    f"**Age:** {age} years"
                )

            else:

                st.write(
                    "**Age:** Face not detected"
                )

        if nationality in [
            "Indian",
            "African"
        ]:

            st.write(
                f"**Dress Colour:** {dress_colour}"
            )


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()