from fastapi import FastAPI
import requests
import tensorflow.lite as tflite
import numpy as np
from PIL import Image
from io import BytesIO
import os
from dotenv import load_dotenv
from mangum import Mangum

# Load environment variables from .env file
load_dotenv()

# Load TFLite Model
interpreter = tflite.Interpreter(model_path="multi_class_satellite_model.tflite")
interpreter.allocate_tensors()

# Get input/output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_names = ["Forest", "Pasture", "River", "AnnualCrop", "Industrial", "SeaLake"]

app = FastAPI()

# Function to download satellite image from Mapbox API
def get_satellite_image(north, south, east, west, zoom):
    mapbox_token = os.getenv("MAPBOX_ACCESS_TOKEN")

    # Compute center latitude & longitude
    center_lat = (north + south) / 2
    center_lon = (east + west) / 2

    url = f"https://api.mapbox.com/styles/v1/mapbox/satellite-v9/static/{center_lon},{center_lat},{zoom}/400x400?access_token={mapbox_token}"

    response = requests.get(url)

    if response.status_code != 200:
        raise Exception(f"Error fetching image: {response.status_code}, {response.text}")

    try:
        img = Image.open(BytesIO(response.content))
        return img
    except Exception as e:
        raise Exception(f"Failed to process image: {e}")

# Function to make predictions using TFLite model
def predict_image(img):
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]

    return {class_names[i]: float(predictions[i]) for i in range(len(class_names))}

# API Route for Predictions
@app.post("/predict/")
async def predict(north: float, south: float, east: float, west: float, zoom: int):
    img = get_satellite_image(north, south, east, west, zoom)
    probabilities = predict_image(img)
    return {"predictions": probabilities}

@app.get("/")
def root():
    return {"message": "Satellite Image Classifier API is running! Use /predict/ to classify images."}

# AWS Lambda handler
handler = Mangum(app)
