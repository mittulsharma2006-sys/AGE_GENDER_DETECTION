import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from insightface.app import FaceAnalysis

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Long Hair Identification",
    page_icon="💇",
    layout="centered"
)

st.title("💇 Long Hair Identification")
st.write(
    "Upload an image to detect age, gender, hair length, "
    "and apply the Task 4 gender logic."
)


# -----------------------------
# Load hair model
# -----------------------------
@st.cache_resource
def load_hair_model():
    return tf.keras.models.load_model(
        "models/hair_length_model.keras"
    )


hair_model = load_hair_model()


# -----------------------------
# Load InsightFace
# -----------------------------
@st.cache_resource
def load_face_model():
    face_app = FaceAnalysis(name="buffalo_l")
    face_app.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )
    return face_app


face_app = load_face_model()


# -----------------------------
# Hair-region extraction
# -----------------------------
def get_hair_region(image_bgr, bbox):
    """
    Create a region around the detected head.
    The region includes space above and around
    the face so longer hair can be captured.
    """

    height, width = image_bgr.shape[:2]

    x1, y1, x2, y2 = bbox.astype(int)

    face_width = x2 - x1
    face_height = y2 - y1

    # Expand region around the face.
    # More area is included above and below
    # to capture long hair.
    new_x1 = max(0, int(x1 - 0.75 * face_width))
    new_x2 = min(width, int(x2 + 0.75 * face_width))

    new_y1 = max(0, int(y1 - 1.25 * face_height))
    new_y2 = min(height, int(y2 + 2.0 * face_height))

    crop = image_bgr[new_y1:new_y2, new_x1:new_x2]

    return crop


# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)


if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.subheader("Uploaded Image")
    st.image(image, use_container_width=True)

    if st.button("Analyze Image"):

        # Convert PIL → OpenCV
        image_np = np.array(image)

        image_bgr = cv2.cvtColor(
            image_np,
            cv2.COLOR_RGB2BGR
        )

        # -----------------------------
        # Detect face
        # -----------------------------
        faces = face_app.get(image_bgr)

        if len(faces) == 0:
            st.error("No face detected in the image.")
            st.stop()

        # Use largest face
        face = max(
            faces,
            key=lambda f: (
                f.bbox[2] - f.bbox[0]
            ) * (
                f.bbox[3] - f.bbox[1]
            )
        )

        # -----------------------------
        # Age
        # -----------------------------
        age = int(round(face.age))

        # -----------------------------
        # Gender
        # -----------------------------
        detected_gender = (
            "Male"
            if face.gender == 1
            else "Female"
        )

        # -----------------------------
        # Extract hair region
        # -----------------------------
        hair_region = get_hair_region(
            image_bgr,
            face.bbox
        )

        if hair_region.size == 0:
            st.error("Could not extract the hair region.")
            st.stop()

        # Convert BGR → RGB
        hair_region_rgb = cv2.cvtColor(
            hair_region,
            cv2.COLOR_BGR2RGB
        )

        # Display hair region
        st.subheader("Detected Hair Region")
        st.image(
            hair_region_rgb,
            use_container_width=True
        )

        # -----------------------------
        # Hair prediction
        # -----------------------------
        hair_input = cv2.resize(
            hair_region_rgb,
            (224, 224)
        )

        hair_input = (
            hair_input.astype(np.float32)
        )

        hair_input = np.expand_dims(
            hair_input,
            axis=0
        )

        hair_prediction = hair_model.predict(
            hair_input,
            verbose=0
        )[0][0]

        # Classes:
        # 0 = long
        # 1 = short
        if hair_prediction >= 0.5:
            hair_length = "Short"
            hair_confidence = float(hair_prediction)
        else:
            hair_length = "Long"
            hair_confidence = 1.0 - float(hair_prediction)

        # -----------------------------
        # Task 4 logic
        # -----------------------------
        if 20 <= age <= 30:

            if hair_length == "Long":
                final_gender = "Female"
            else:
                final_gender = "Male"

            rule_applied = True

        else:

            final_gender = detected_gender
            rule_applied = False

        # -----------------------------
        # Results
        # -----------------------------
        st.subheader("Results")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Age",
                f"{age} years"
            )

            st.metric(
                "Detected Gender",
                detected_gender
            )

        with col2:
            st.metric(
                "Hair Length",
                hair_length
            )

            st.metric(
                "Final Gender",
                final_gender
            )

        st.write(
            f"Hair prediction confidence: "
            f"{hair_confidence * 100:.2f}%"
        )

        if rule_applied:

            st.success(
                "Task 4 rule applied: "
                "for ages 20–30, hair length determines "
                "the final gender."
            )

        else:

            st.info(
                "Task 4 hair-based rule was not applied "
                "because the detected age is outside 20–30."
            )