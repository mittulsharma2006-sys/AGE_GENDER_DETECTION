from ultralytics import YOLO
import cv2

model = YOLO("yolo11n.pt")

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

print("Webcam started. Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    results = model.track(
        frame,
        persist=True,
        classes=[0],
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow("Task 6 - Person Detection", annotated_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()