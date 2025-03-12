# Satellite Classifier - Backend API

This repository contains the backend API and machine learning model used to classify satellite imagery into six land types: **Forest**, **Pasture**, **River**, **Crops**, **Urban / Industrial**, and **Sea / Lake**.

The API is powered by a Convolutional Neural Network (CNN) trained on a multi-class satellite dataset and served via FastAPI. It is hosted on **Render.com**.

The repository for the frontend can be found here: https://github.com/asmerdon/satellite-classifier-frontend

**Working live at:** https://asmerdon.github.io/satellite-classifier-frontend/

## Model Overview
The model was trained on the ["Trees in Satellite Imagery"](https://www.kaggle.com/datasets/mcagriaksoy/trees-in-satellite-imagery) dataset available on Kaggle. The dataset includes labeled images for six classes:
- Forest  
- Pasture  
- River  
- AnnualCrop  
- Industrial  
- SeaLake  

### Preprocessing and Training Techniques:
- Images were resized to **128x128x3**.
- **Data augmentation** was applied, including random flips, rotations, brightness adjustment, and zoom.
- **Class weights** were used during training to account for class imbalance across land types.
- The model uses a **CNN architecture** with convolutional layers, max pooling layers, a dense layer, dropout, and softmax output.


## Backend Overview

The backend API is built using **FastAPI**. It accepts user requests containing geographic bounds and a zoom level, then fetches a satellite image using the **Mapbox Static API**. The image is processed and passed to the trained CNN model for classification.

The API returns softmax-based probabilities for each of the six land types.


## API Endpoint

### `POST /predict/`

**Query parameters:**
- `north` (float)
- `south` (float)
- `east` (float)
- `west` (float)
- `zoom` (int)

Returns: A JSON response with prediction probabilities for each land type.


## Hosting & Deployment

- The backend is hosted on **Render.com** using a free-tier web service.
- A `render.yaml` file is included for Render deployment configuration.
- Due to Render's free-tier behaviour, the **first request may take up to 50 seconds** to respond while the server spins up.


## Environment Variables

A `.env` file (excluded from GitHub) is used to load environment variables such as:

- `MAPBOX_ACCESS_TOKEN`

## Requirements

The backend uses the following Python packages:

- `fastapi`  
- `uvicorn`  
- `requests`  
- `pillow`  
- `numpy`  
- `python-dotenv`  
- `mangum`  
- `tensorflow`


