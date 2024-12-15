from main import ModelSegment

model = ModelSegment(image_path="test_image.jpg")
res = model.segment_tornlist()
print(res)