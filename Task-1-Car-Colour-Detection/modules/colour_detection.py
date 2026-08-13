import cv2
import numpy as np


def detect_car_colour(car_image):
    """
    Detect the approximate dominant colour of a car.

    Returns:
        blue, red, white, black, grey, or other
    """

    if car_image is None or car_image.size == 0:
        return "other"

    # Resize
    image = cv2.resize(car_image, (120, 120))

    h, w = image.shape[:2]

    # Focus more on the lower/body portion of the car.
    # This reduces the effect of windows, sky and background.
    image = image[
        int(h * 0.30):int(h * 0.90),
        int(w * 0.10):int(w * 0.90)
    ]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    total_pixels = image.shape[0] * image.shape[1]

    # --------------------------------
    # BLUE
    # --------------------------------

    blue_mask = (
        (hue >= 90) &
        (hue <= 130) &
        (saturation >= 100) &
        (value >= 50)
    )

    # --------------------------------
    # RED
    # --------------------------------

    red_mask = (
        (
            (hue <= 10) |
            (hue >= 170)
        ) &
        (saturation >= 100) &
        (value >= 50)
    )

    # --------------------------------
    # WHITE
    # --------------------------------

    white_mask = (
        (saturation <= 45) &
        (value >= 170)
    )

    # --------------------------------
    # BLACK
    # --------------------------------

    black_mask = (
        (value <= 65)
    )

    # --------------------------------
    # GREY
    # --------------------------------

    grey_mask = (
        (saturation <= 55) &
        (value > 65) &
        (value < 180)
    )

    # Calculate ratios
    blue_ratio = np.sum(blue_mask) / total_pixels
    red_ratio = np.sum(red_mask) / total_pixels
    white_ratio = np.sum(white_mask) / total_pixels
    black_ratio = np.sum(black_mask) / total_pixels
    grey_ratio = np.sum(grey_mask) / total_pixels

    ratios = {
        "blue": blue_ratio,
        "red": red_ratio,
        "white": white_ratio,
        "black": black_ratio,
        "grey": grey_ratio
    }

    # --------------------------------
    # BLUE / RED REQUIRE STRONG EVIDENCE
    # --------------------------------

    # A blue car should have a significant
    # amount of genuinely saturated blue pixels.

    if blue_ratio >= 0.15:
        return "blue"

    if red_ratio >= 0.15:
        return "red"

    # --------------------------------
    # NEUTRAL COLOURS
    # --------------------------------

    neutral_colour = max(
        {
            "white": white_ratio,
            "black": black_ratio,
            "grey": grey_ratio
        },
        key=lambda x: {
            "white": white_ratio,
            "black": black_ratio,
            "grey": grey_ratio
        }[x]
    )

    neutral_ratio = {
        "white": white_ratio,
        "black": black_ratio,
        "grey": grey_ratio
    }[neutral_colour]

    if neutral_ratio >= 0.15:
        return neutral_colour

    return "other"