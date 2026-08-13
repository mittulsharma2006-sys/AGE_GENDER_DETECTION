import cv2
import os
from ultralytics import YOLO

from .colour_detection import detect_car_colour


class TrafficDetector:

    def __init__(self):

        # YOLO automatically downloads the model
        # the first time it is used.
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),"yolo11n.pt")

        self.model = YOLO(model_path)

        # COCO class IDs
        self.PERSON_CLASS = 0
        self.CAR_CLASS = 2

    def process_image(self, image):

        # Streamlit/PIL uses RGB.
        # OpenCV uses BGR.
        frame = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        results = self.model(
            frame,
            verbose=False
        )

        car_count = 0
        blue_car_count = 0
        other_car_count = 0
        person_count = 0

        for result in results:

            if result.boxes is None:
                continue

            for box in result.boxes:

                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                if confidence < 0.40:
                    continue

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                # -------------------------
                # PERSON
                # -------------------------

                if class_id == self.PERSON_CLASS:

                    person_count += 1

                    # Green rectangle for people
                    box_colour = (0, 255, 0)

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        box_colour,
                        2
                    )

                    cv2.putText(
                        frame,
                        f"Person {confidence:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        box_colour,
                        2
                    )

                # -------------------------
                # CAR
                # -------------------------

                elif class_id == self.CAR_CLASS:

                    car_count += 1

                    x1 = max(0, x1)
                    y1 = max(0, y1)

                    x2 = min(
                        frame.shape[1],
                        x2
                    )

                    y2 = min(
                        frame.shape[0],
                        y2
                    )

                    car_crop = frame[
                        y1:y2,
                        x1:x2
                    ]

                    colour = detect_car_colour(
                        car_crop
                    )

                    # Internship requirement:
                    #
                    # BLUE CAR -> RED BOX
                    # OTHER CAR -> BLUE BOX

                    if colour == "blue":

                        box_colour = (0, 0, 255)

                        blue_car_count += 1

                    else:

                        box_colour = (255, 0, 0)

                        other_car_count += 1

                    cv2.rectangle(
                        frame,
                        (x1, y1),
                        (x2, y2),
                        box_colour,
                        3
                    )

                    cv2.putText(
                        frame,
                        f"Car: {colour}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        box_colour,
                        2
                    )

        # -------------------------
        # SUMMARY
        # -------------------------

        cv2.rectangle(
            frame,
            (10, 10),
            (350, 135),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            frame,
            f"Cars: {car_count}",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Blue Cars: {blue_car_count}",
            (25, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"People: {person_count}",
            (25, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Convert back to RGB
        output = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        statistics = {
            "cars": car_count,
            "blue_cars": blue_car_count,
            "other_cars": other_car_count,
            "people": person_count
        }

        return output, statistics