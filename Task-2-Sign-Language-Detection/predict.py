import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

# =========================
# CONFIGURATION
# =========================

MODEL_PATH = "models/sign_language_model.keras"

CLASSES = ["A", "B", "C", "D", "E", "F"]

# =========================
# LOAD MODEL
# =========================

model = load_model(MODEL_PATH)

# =========================
# MEDIAPIPE SETUP
# =========================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# =========================
# IMAGE PATH
# =========================

image_path = "dataset/F/F100.jpg"

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")

# =========================
# EXTRACT LANDMARKS
# =========================

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = hands.process(image_rgb)

if not results.multi_hand_landmarks:
    print("No hand detected in the image.")
    hands.close()
    exit()

hand = results.multi_hand_landmarks[0]

landmarks = []

for landmark in hand.landmark:
    landmarks.extend([
        landmark.x,
        landmark.y,
        landmark.z
    ])

# =========================
# PREDICTION
# =========================

X = np.array(landmarks, dtype=np.float32).reshape(1, -1)

prediction = model.predict(X, verbose=0)

predicted_index = np.argmax(prediction[0])

predicted_class = CLASSES[predicted_index]

confidence = prediction[0][predicted_index] * 100

# =========================
# RESULT
# =========================

print("\n==============================")
print("SIGN LANGUAGE PREDICTION")
print("==============================")

print(f"Image: {image_path}")
print(f"Predicted Sign: {predicted_class}")
print(f"Confidence: {confidence:.2f}%")

print("==============================")

hands.close()