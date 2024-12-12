import cv2
from ultralytics import YOLO

model = YOLO("best.pt")

image_path = "test_image.jpg" 

try:
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    results = model.predict(img)

    annotated_image = results[0].plot()

    desired_width = 1080
    desired_height = 720
    annotated_image = cv2.resize(annotated_image, (desired_width, desired_height))
    cv2.imshow("YOLO11 Inference", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

except FileNotFoundError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")