import cv2
from ultralytics import YOLO

# class ModelSegment:
#     def __init__(self, pt = "best.pt", image_path = "test_image.jpg"):
#         self.best_w = pt
#         self.image_path = image_path

#     def segment_tornlist(self):
#         model = YOLO(self.best_w)

#         try:
#             img = cv2.imread(self.image_path)
#             if img is None:
#                 raise FileNotFoundError(f"Image not found at {self.image_path}")

#             results = model.predict(source = img, conf = 0.30)

#             annotated_image = results[0].plot()

#             desired_width = 1080
#             desired_height = 720
#             annotated_image = cv2.resize(annotated_image, (desired_width, desired_height))
#             cv2.imshow("YOLO11 Inference", annotated_image)
#             cv2.waitKey(0)
#             cv2.destroyAllWindows()
            
#         except FileNotFoundError as e:
#             print(f"Error: {e}")

#         except Exception as e:
            # print(f"An unexpected error occurred: {e}")

class ModelSegment:
    def __init__(self, pt="best.pt", image_path="test_image.jpg"):
        self.best_w = pt
        self.image_path = image_path

    def segment_tornlist(self):
        model = YOLO(self.best_w)

        try:
            img = cv2.imread(self.image_path)
            if img is None:
                raise FileNotFoundError(f"Image not found at {self.image_path}")

            results = model.predict(source=img, conf=0.30)

            # Получаем классы обнаруженных объектов
            detected_classes = results[0].boxes.cls

            # Проверяем, есть ли класс "torn" (предполагая, что "torn" имеет индекс, например, 0)
            # Вам нужно будет заменить 0 на фактический индекс класса "torn" в вашей модели
            if any(cls == 0 for cls in detected_classes):  # Замените 0 на индекс класса "torn"
                print("да")
            else:
                print("нет")

            annotated_image = results[0].plot()
            desired_width = 1080
            desired_height = 720
            annotated_image = cv2.resize(annotated_image, (desired_width, desired_height))
            cv2.imshow("YOLO Inference", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except FileNotFoundError as e:
            print(f"Error: {e}")

        except Exception as e:
            print(f"An unexpected error occurred: {e}")

