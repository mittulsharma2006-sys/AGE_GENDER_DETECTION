import os
import csv
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
from ultralytics import YOLO
from insightface.app import FaceAnalysis


# =========================
# CONFIGURATION
# =========================

YOLO_MODEL = "yolo11n.pt"
AGE_GENDER_MODEL = "models/Age_Sex_Detection.h5"
OUTPUT_DIR = "output"
LOG_FILE = os.path.join(OUTPUT_DIR, "senior_citizen_log.csv")

CAMERA_INDEX = 1


# =========================
# CREATE OUTPUT DIRECTORY
# =========================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# LOAD MODELS
# =========================

print("Loading YOLO...")
yolo = YOLO(YOLO_MODEL)

print("Loading age/gender model...")
age_gender_model = tf.keras.models.load_model(
    AGE_GENDER_MODEL,
    compile=False
)

print("Loading InsightFace...")

face_app = FaceAnalysis(
    name="buffalo_l",
    providers=["CPUExecutionProvider"]
)

face_app.prepare(
    ctx_id=0,
    det_size=(640, 640)
)

print("All models loaded successfully.")


# =========================
# CREATE CSV FILE
# =========================

if not os.path.exists(LOG_FILE):

    with open(LOG_FILE, "w", newline="", encoding="utf-8") as file:

        writer = csv.writer(file)

        writer.writerow([
            "Person ID",
            "Age",
            "Gender",
            "Senior Citizen",
            "Time of Visit"
        ])


# =========================
# TRACKING LOG
# =========================

logged_persons = set()


# =========================
# OPEN WEBCAM
# =========================

cap = cv2.VideoCapture(
    CAMERA_INDEX,
    cv2.CAP_DSHOW
)

if not cap.isOpened():

    print("Could not open webcam.")

    exit()


print("Webcam started.")
print("Press Q to quit.")


# =========================
# MAIN LOOP
# =========================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Could not read frame.")

        break


    # -------------------------
    # YOLO PERSON TRACKING
    # -------------------------

    results = yolo.track(
        frame,
        persist=True,
        classes=[0],
        verbose=False
    )


    boxes = results[0].boxes


    if boxes is not None and len(boxes) > 0:

        # Get tracking IDs
        if boxes.id is not None:

            track_ids = boxes.id.int().cpu().tolist()

        else:

            track_ids = list(range(len(boxes)))


        # -------------------------
        # PROCESS EACH PERSON
        # -------------------------

        for box, person_id in zip(boxes, track_ids):

            x1, y1, x2, y2 = box.xyxy[0].int().cpu().tolist()

            # Keep coordinates inside frame
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)


            person_crop = frame[y1:y2, x1:x2]


            if person_crop.size == 0:

                continue


            # -------------------------
            # FACE DETECTION
            # -------------------------

            faces = face_app.get(person_crop)


            if len(faces) == 0:

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Person {person_id} - Face not detected",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 0),
                    2
                )

                continue


            # Select largest face
            face = max(
                faces,
                key=lambda f: (
                    f.bbox[2] - f.bbox[0]
                ) * (
                    f.bbox[3] - f.bbox[1]
                )
            )


            fx1, fy1, fx2, fy2 = face.bbox.astype(int)


            # Keep face coordinates inside person crop
            fx1 = max(0, fx1)
            fy1 = max(0, fy1)
            fx2 = min(person_crop.shape[1], fx2)
            fy2 = min(person_crop.shape[0], fy2)


            face_crop = person_crop[
                fy1:fy2,
                fx1:fx2
            ]


            if face_crop.size == 0:

                continue


            # -------------------------
            # PREPARE FACE
            # -------------------------

            face_rgb = cv2.cvtColor(
                face_crop,
                cv2.COLOR_BGR2RGB
            )

            face_rgb = cv2.resize(
                face_rgb,
                (48, 48)
            )

            face_input = (
                face_rgb.astype(np.float32) / 255.0
            )

            face_input = np.expand_dims(
                face_input,
                axis=0
            )


            # -------------------------
            # AGE + GENDER PREDICTION
            # -------------------------

            predictions = age_gender_model.predict(
                face_input,
                verbose=0
            )


            gender_prediction = float(
                predictions[0][0][0]
            )

            age_prediction = float(
                predictions[1][0][0]
            )


            age = int(
                round(age_prediction)
            )

            age = max(
                1,
                min(100, age)
            )


            if gender_prediction >= 0.5:

                gender = "Male"

            else:

                gender = "Female"


            # -------------------------
            # SENIOR CITIZEN RULE
            # -------------------------

            if age > 60:

                senior = "Yes"

                box_color = (0, 0, 255)

            else:

                senior = "No"

                box_color = (0, 255, 0)


            # -------------------------
            # LOG VISIT
            # -------------------------

            if person_id not in logged_persons:

                visit_time = datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                )


                with open(
                    LOG_FILE,
                    "a",
                    newline="",
                    encoding="utf-8"
                ) as file:

                    writer = csv.writer(file)

                    writer.writerow([
                        person_id,
                        age,
                        gender,
                        senior,
                        visit_time
                    ])


                logged_persons.add(person_id)

                print(
                    f"Logged Person {person_id}: "
                    f"{age}, {gender}, "
                    f"Senior={senior}, "
                    f"{visit_time}"
                )


            # -------------------------
            # DISPLAY RESULT
            # -------------------------

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                box_color,
                2
            )


            label = (
                f"ID {person_id} | "
                f"{gender} | "
                f"Age {age}"
            )


            if senior == "Yes":

                label += " | SENIOR"


            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2
            )


    # -------------------------
    # SHOW FRAME
    # -------------------------

    cv2.imshow(
        "Task 6 - Senior Citizen Identification",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# =========================
# CLEANUP
# =========================

cap.release()

cv2.destroyAllWindows()

print("\nApplication closed.")

print(
    f"Visit log saved to: {LOG_FILE}"
)