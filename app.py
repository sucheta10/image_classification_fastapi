import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import cv2
import pathlib

app = FastAPI()

# Load model
model = tf.keras.models.load_model(
    'C:/Users/Sucheta/OneDrive/Desktop/python/image_classification_fast_api/my_model2.hdf5'
)

# Compile model if necessary (usually required if you want to use model metrics)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load class names
data_dir = pathlib.Path("C:/Users/Sucheta/OneDrive/Desktop/python/100 sports image classification/train")
class_names = sorted([item.name for item in data_dir.glob('*') if item.is_dir()])


def import_and_predict(image_data, model):
    size = (180, 180)  # Resize the image to the input size expected by the model
    image = ImageOps.fit(image_data, size, Image.LANCZOS)  # Resize and maintain aspect ratio
    image = np.asarray(image)  # Convert to numpy array
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    img_reshape = img[np.newaxis, ...]  # Reshape to match model's input shape
    prediction = model.predict(img_reshape)  # Get prediction
    return prediction


@app.get("/")
async def read_root():
    return {"message": "Welcome to the sports image classification API!"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image file
        image = Image.open(file.file)

        # Make prediction
        predictions = import_and_predict(image, model)
        score = tf.nn.softmax(predictions[0])
        predicted_class = class_names[np.argmax(score)]
        confidence = 100 * np.max(score)

        return JSONResponse(content={"prediction": predicted_class, "confidence": confidence})

    except Exception as e:
        return JSONResponse(content={"error": str(e)})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
