import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from PIL import Image
from datetime import datetime
from streamlit_webrtc import webrtc_streamer
import av
# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/sign_language_model.keras"

CLASSES = ["A", "B", "C", "D", "E", "F"]

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Sign Language Detection",
    page_icon="🤟",
    layout="wide"
)

# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #ddd;
    margin-top: 20px;
}

.sign {
    font-size: 60px;
    font-weight: bold;
}

.confidence {
    font-size: 22px;
}

</style>
""", unsafe_allow_html=True)

# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">🤟 Sign Language Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">AI-powered recognition of ASL signs A–F</div>',
    unsafe_allow_html=True
)

# ============================================================
# TIME RESTRICTION
# ============================================================

current_time = datetime.now().time()

start_time = datetime.strptime("18:00", "%H:%M").time()
end_time = datetime.strptime("22:00", "%H:%M").time()

if not (start_time <= current_time <= end_time):

    st.warning(
        "⏰ This application is operational only between "
        "6:00 PM and 10:00 PM."
    )

    st.info(
        f"Current system time: {datetime.now().strftime('%I:%M %p')}"
    )

    st.stop()

# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_sign_model():
    return load_model(MODEL_PATH)


model = load_sign_model()

# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands_static = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

hands_video = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ============================================================
# LANDMARK EXTRACTION
# ============================================================

def extract_landmarks(results):

    if not results.multi_hand_landmarks:
        return None

    hand = results.multi_hand_landmarks[0]

    landmarks = []

    for landmark in hand.landmark:

        landmarks.extend([
            landmark.x,
            landmark.y,
            landmark.z
        ])

    return np.array(
        landmarks,
        dtype=np.float32
    ).reshape(1, -1)


# ============================================================
# PREDICTION
# ============================================================

def predict_sign(landmarks):

    prediction = model.predict(
        landmarks,
        verbose=0
    )[0]

    index = np.argmax(prediction)

    sign = CLASSES[index]

    confidence = prediction[index] * 100

    return sign, confidence


# ============================================================
# TABS
# ============================================================

tab1, tab2 = st.tabs(
    ["📤 Upload Image", "🎥 Real-Time Video"]
)

# ============================================================
# IMAGE UPLOAD
# ============================================================

with tab1:

    st.header("Upload an Image")

    uploaded_file = st.file_uploader(
        "Choose an image containing a hand sign",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                image,
                caption="Uploaded Image",
                
            )

        image_array = np.array(image)

        results = hands_static.process(image_array)

        landmarks = extract_landmarks(results)

        with col2:

            if landmarks is None:

                st.error(
                    "❌ No hand detected in the image."
                )

            else:

                sign, confidence = predict_sign(
                    landmarks
                )

                st.markdown(
                    f"""
                    <div class="result-box">

                    <div>Predicted Sign</div>

                    <div class="sign">
                    {sign}
                    </div>

                    <div class="confidence">
                    Confidence: {confidence:.2f}%
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

# ============================================================
# REAL-TIME VIDEO
# ============================================================

with tab2:

    st.header("🎥 Real-Time Sign Language Detection")

    st.write(
        "Show your hand to the camera. "
        "The model will continuously detect signs A, B, C, D, E and F."
    )

    def video_frame_callback(frame):

        img = frame.to_ndarray(format="bgr24")

        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # Detect hand
        results = hands_video.process(
            rgb_frame
        )

        # If hand detected
        if results.multi_hand_landmarks:

            landmarks = extract_landmarks(
                results
            )

            if landmarks is not None:

                sign, confidence = predict_sign(
                    landmarks
                )

                # Draw hand landmarks
                for hand_landmarks in results.multi_hand_landmarks:

                    mp_drawing.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                    )

                # Display prediction on video
                cv2.rectangle(
                    img,
                    (10, 10),
                    (360, 100),
                    (0, 0, 0),
                    -1
                )

                cv2.putText(
                    img,
                    f"Sign: {sign}",
                    (25, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    img,
                    f"Confidence: {confidence:.1f}%",
                    (25, 85),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

        else:

            cv2.putText(
                img,
                "No hand detected",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2
            )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )

    webrtc_streamer(
        key="sign-language-camera",
        video_frame_callback=video_frame_callback,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )
# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Sign Language Detection | Machine Learning Internship Task 2"
)