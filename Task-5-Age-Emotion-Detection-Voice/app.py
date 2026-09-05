import io
import os

import joblib
import librosa
import numpy as np
import soundfile as sf
import streamlit as st


# ============================================================
# MODEL FILES
# ============================================================

GENDER_MODEL = "models/gender_voice_model.pkl"
AGE_MODEL = "models/age_regression_model.pkl"
SENIOR_MODEL = "models/senior_voice_model.pkl"
EMOTION_MODEL = "models/emotion_svm_model.pkl"
EMOTION_LABELS = "models/emotion_svm_labels.npy"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Age & Emotion Detection",
    page_icon="🎙️",
    layout="centered"
)


# ============================================================
# AUDIO FEATURE EXTRACTION
# ============================================================

def extract_features(audio_bytes):

    audio, sr = sf.read(
        io.BytesIO(audio_bytes),
        dtype="float32"
    )

    # Stereo → Mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Resample → 16 kHz
    if sr != 16000:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=16000
        )

    # Exactly 4 seconds
    target_length = 16000 * 4

    if len(audio) < target_length:

        audio = np.pad(
            audio,
            (0, target_length - len(audio))
        )

    else:

        audio = audio[:target_length]

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=16000,
        n_mfcc=40
    )

    # Delta
    delta = librosa.feature.delta(
        mfcc
    )

    # Delta-Delta
    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    # 240 features
    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),

        np.mean(delta, axis=1),
        np.std(delta, axis=1),

        np.mean(delta2, axis=1),
        np.std(delta2, axis=1)
    ])

    return features.reshape(1, -1)


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    gender_model = joblib.load(
        GENDER_MODEL
    )

    age_model = joblib.load(
        AGE_MODEL
    )

    senior_model = joblib.load(
        SENIOR_MODEL
    )

    emotion_model = joblib.load(
        EMOTION_MODEL
    )

    emotion_labels = np.load(
        EMOTION_LABELS,
        allow_pickle=True
    )

    return (
        gender_model,
        age_model,
        senior_model,
        emotion_model,
        emotion_labels
    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "🎙️ Age & Emotion Detection Through Voice"
)

st.write(
    "Upload a voice recording to analyze gender, "
    "approximate age, senior-citizen status, and emotion."
)

st.info(
    "This system is designed to process male voices only."
)


# ============================================================
# CHECK MODEL FILES
# ============================================================

required_models = [
    GENDER_MODEL,
    AGE_MODEL,
    SENIOR_MODEL,
    EMOTION_MODEL,
    EMOTION_LABELS
]

missing_models = [
    model
    for model in required_models
    if not os.path.exists(model)
]

if missing_models:

    st.error(
        "Some required model files are missing:"
    )

    for model in missing_models:
        st.write(
            f"- `{model}`"
        )

    st.stop()


# ============================================================
# LOAD MODELS
# ============================================================

try:

    (
        gender_model,
        age_model,
        senior_model,
        emotion_model,
        emotion_labels
    ) = load_models()

except Exception as e:

    st.error(
        f"Could not load the models: {e}"
    )

    st.stop()


# ============================================================
# AUDIO UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a voice recording",
    type=[
        "wav",
        "flac",
        "ogg",
        "mp3"
    ]
)


# ============================================================
# PROCESS AUDIO
# ============================================================

if uploaded_file is not None:

    st.audio(
        uploaded_file
    )

    st.write(
        f"Selected file: **{uploaded_file.name}**"
    )

    if st.button(
        "🔍 Analyze Voice",
        use_container_width=True
    ):

        try:

            # ------------------------------------------------
            # Read audio
            # ------------------------------------------------

            audio_bytes = uploaded_file.read()

            if len(audio_bytes) == 0:

                st.error(
                    "The uploaded audio file is empty."
                )

                st.stop()


            # ------------------------------------------------
            # Extract features
            # ------------------------------------------------

            with st.spinner(
                "Extracting audio features..."
            ):

                features = extract_features(
                    audio_bytes
                )


            # =================================================
            # GENDER DETECTION
            # =================================================

            gender_prediction = (
                gender_model.predict(
                    features
                )[0]
            )

            gender_probability = (
                gender_model.predict_proba(
                    features
                )[0]
            )

            gender_confidence = (
                np.max(
                    gender_probability
                ) * 100
            )


            if gender_prediction == 1:

                detected_gender = "Male"

            else:

                detected_gender = "Female"


            # =================================================
            # FEMALE REJECTION
            # =================================================

            if detected_gender == "Female":

                st.error(
                    "Upload male voice."
                )

                st.write(
                    "Detected Gender: **Female**"
                )

                st.write(
                    f"Gender Confidence: "
                    f"**{gender_confidence:.2f}%**"
                )

                st.stop()


            # =================================================
            # MALE VOICE
            # =================================================

            st.success(
                f"Male voice detected "
                f"({gender_confidence:.2f}% confidence)"
            )


            # =================================================
            # AGE PREDICTION
            # =================================================

            with st.spinner(
                "Predicting approximate age..."
            ):

                predicted_age = (
                    age_model.predict(
                        features
                    )[0]
                )


            # Keep within reasonable range
            predicted_age = max(
                1,
                min(
                    100,
                    predicted_age
                )
            )

            predicted_age = round(
                predicted_age
            )


            st.subheader(
                "🎂 Age Prediction"
            )

            st.metric(
                "Approximate Age",
                f"{predicted_age} years"
            )


            # =================================================
            # SENIOR CITIZEN CLASSIFIER
            # =================================================

            senior_prediction = (
                senior_model.predict(
                    features
                )[0]
            )

            senior_probability = (
                senior_model.predict_proba(
                    features
                )[0]
            )

            # Class 1 = 60+
            senior_60_probability = (
                senior_probability[1] * 100
            )


            # =================================================
            # FINAL SENIOR DECISION
            # =================================================
            #
            # Primary condition:
            # predicted age > 60
            #
            # Supporting condition:
            # senior classifier probability >= 75%
            #
            # =================================================

            is_senior = (
                predicted_age > 60
                or
                senior_60_probability >= 75
            )


            if is_senior:

                # =================================================
                # SENIOR CITIZEN
                # =================================================

                st.warning(
                    "👴 Senior Citizen"
                )

                if predicted_age > 60:

                    st.write(
                        "Predicted age is above 60 years."
                    )

                else:

                    st.write(
                        "The senior-citizen classifier "
                        "strongly indicates a 60+ voice."
                    )

                    st.write(
                        f"60+ Classifier Probability: "
                        f"**{senior_60_probability:.2f}%**"
                    )


                # =================================================
                # EMOTION DETECTION
                # =================================================

                with st.spinner(
                    "Detecting emotion..."
                ):

                    emotion_prediction = (
                        emotion_model.predict(
                            features
                        )[0]
                    )

                    emotion_probability = (
                        emotion_model.predict_proba(
                            features
                        )[0]
                    )

                    emotion_confidence = (
                        np.max(
                            emotion_probability
                        ) * 100
                    )


                # RAVDESS emotion mapping
                emotion_map = {

                    "01": "Neutral",

                    "02": "Calm",

                    "03": "Happy",

                    "04": "Sad",

                    "05": "Angry",

                    "06": "Fearful",

                    "07": "Disgust",

                    "08": "Surprised"
                }


                raw_emotion = str(
                    emotion_labels[
                        int(emotion_prediction)
                    ]
                )

                emotion_name = emotion_map.get(
                    raw_emotion,
                    raw_emotion
                )


                st.subheader(
                    "😊 Emotion Detection"
                )

                st.success(
                    f"Detected Emotion: "
                    f"**{emotion_name}**"
                )

                st.write(
                    f"Emotion Confidence: "
                    f"**{emotion_confidence:.2f}%**"
                )


            else:

                # =================================================
                # UNDER 60
                # =================================================

                st.info(
                    "The person is classified "
                    "as under 60."
                )

                st.caption(
                    "Emotion detection is only performed "
                    "for individuals above 60 years of age "
                    "according to the task requirements."
                )


        except Exception as e:

            st.error(
                f"Error processing audio: {e}"
            )