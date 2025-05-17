import keras_ocr

# Create pipeline
pipeline = keras_ocr.pipeline.Pipeline()

# path to an image file
image_path = r"C:\Users\DELL\Downloads\610f4b5b-65ef-43b5-8569-d7e2efcc72b7.jpg"
images = [keras_ocr.tools.read(image_path)]

# Run OCR
prediction_groups = pipeline.recognize(images)

for text, box in prediction_groups[0]:
    print(f'Detected text: {text}, at: {box}')
