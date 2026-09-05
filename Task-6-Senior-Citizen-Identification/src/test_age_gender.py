import cv2
import numpy as np
import tensorflow as tf
from insightface.app import FaceAnalysis

# Load age/gender model
model = tf.keras.models.load_model(
    "models/Age_Sex_Detection.h5",
    compile=False
)

# Load InsightFace
face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(ctx_id=0, det_size=(640, 640))

# Open webcam
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("Webcam started. Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Could not read frame")
        break

    faces = face_app.get(frame)

    for face in faces:

        x1, y1, x2, y2 = face.bbox.astype(int)

        # Keep coordinates inside image
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        # Convert BGR → RGB
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

        # Resize to model input
        face_rgb = cv2.resize(face_rgb, (48, 48))

        # Normalize
        face_input = face_rgb.astype(np.float32) / 255.0

        # Add batch dimension
        face_input = np.expand_dims(face_input, axis=0)

        # Predict
        predictions = model.predict(face_input, verbose=0)

        gender_prediction = predictions[0][0][0]
        age_prediction = predictions[1][0][0]

        # Gender
        if gender_prediction >= 0.5:
            gender = "Male"
        else:
            gender = "Female"

        age = int(round(float(age_prediction)))

        # Senior citizen rule
        if age > 60:
            status = "Senior Citizen"
        else:
            status = "Not Senior"

        # Draw box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"{gender}, Age: {age}"

        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            status,
            (x1, y2 + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255) if age > 60 else (255, 255, 255),
            2
        )

    cv2.imshow("Task 6 - Age and Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()