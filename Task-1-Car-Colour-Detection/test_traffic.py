import cv2

from modules.traffic_detection import TrafficDetector


image = cv2.imread(
    "sample_images/traffic.jpg"
)

if image is None:
    print("ERROR: traffic.jpg not found")
    exit()


image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


detector = TrafficDetector()


result, statistics = detector.process_image(
    image_rgb
)


print("\nDetection Results")
print("-----------------")

print(
    "Cars:",
    statistics["cars"]
)

print(
    "Blue Cars:",
    statistics["blue_cars"]
)

print(
    "Other Cars:",
    statistics["other_cars"]
)

print(
    "People:",
    statistics["people"]
)


result_bgr = cv2.cvtColor(
    result,
    cv2.COLOR_RGB2BGR
)


cv2.imwrite(
    "traffic_result.jpg",
    result_bgr
)


print("\nResult saved as traffic_result.jpg")