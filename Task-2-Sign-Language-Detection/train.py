import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

# =========================
# CONFIGURATION
# =========================

DATASET_DIR = "dataset"
MODEL_DIR = "models"

CLASSES = ["A", "B", "C", "D", "E", "F"]

IMAGE_LIMIT_PER_CLASS = 500

os.makedirs(MODEL_DIR, exist_ok=True)

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
# EXTRACT HAND LANDMARKS
# =========================

def extract_landmarks(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = hands.process(image_rgb)

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

    return landmarks


# =========================
# LOAD DATASET
# =========================

X = []
y = []

print("\nLoading dataset...")
print("=" * 50)

for class_index, class_name in enumerate(CLASSES):

    class_path = os.path.join(DATASET_DIR, class_name)

    image_files = [
        f for f in os.listdir(class_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    image_files = image_files[:IMAGE_LIMIT_PER_CLASS]

    print(f"{class_name}: {len(image_files)} images")

    detected = 0

    for filename in image_files:

        image_path = os.path.join(class_path, filename)

        landmarks = extract_landmarks(image_path)

        if landmarks is not None:

            X.append(landmarks)
            y.append(class_index)

            detected += 1

    print(f"  Hand detected: {detected}")


hands.close()

# =========================
# CONVERT DATA
# =========================

X = np.array(X, dtype=np.float32)
y = np.array(y)

print("\nDataset prepared!")
print("Samples:", len(X))
print("Features:", X.shape[1] if len(X) > 0 else 0)

if len(X) == 0:
    raise RuntimeError(
        "No hands were detected. Check the dataset and MediaPipe installation."
    )

# =========================
# TRAIN / TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

y_train = to_categorical(y_train, num_classes=len(CLASSES))
y_test = to_categorical(y_test, num_classes=len(CLASSES))

# =========================
# BUILD MODEL
# =========================

model = Sequential([
    Dense(128, activation="relu", input_shape=(63,)),
    Dropout(0.3),

    Dense(64, activation="relu"),
    Dropout(0.2),

    Dense(32, activation="relu"),

    Dense(len(CLASSES), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# =========================
# TRAIN MODEL
# =========================

print("\nTraining model...")
print("=" * 50)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    verbose=1
)

# =========================
# EVALUATE MODEL
# =========================

loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print("\nModel Evaluation")
print("=" * 50)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# =========================
# SAVE MODEL
# =========================

model_path = os.path.join(
    MODEL_DIR,
    "sign_language_model.keras"
)

model.save(model_path)

print("\nModel saved successfully!")
print(f"Location: {model_path}")