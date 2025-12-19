# Code-Alchemy-Startup
# NEXUS – AI Road Intelligence Platform 🚧🤖

## Overview
NEXUS is an AI-based pothole detection and road monitoring system built using a YOLO object detection model.  
The project detects potholes from road images, generates annotated outputs, and visualizes insights through an interactive dashboard.

The system integrates **deep learning inference**, **analytics**, and **visual reporting** to demonstrate how AI can assist in smart road infrastructure monitoring.

---

## Features
- Pothole detection using YOLO
- Bounding box visualization with confidence scores
- Road Quality Score (RQS) computation
- Interactive Streamlit dashboard
- Analytics and geospatial visualization
- Professional frontend interface (NEXUS UI)

---

## Dataset
This project uses a **public pothole detection dataset** containing annotated road images.

- Dataset type: Road surface images
- Annotation format: YOLO bounding boxes
- Task: Object detection (pothole)

The full dataset is **not included** in this repository due to size constraints.  
Sample input and output images are provided to demonstrate results.

---

## Model
- Model: YOLOv8
- Framework: Ultralytics YOLO
- Task: Pothole detection

⚠️ Trained model weights (`.pt`) are not included in this repository.

---

## Project Structure

NEXUS/
├── app.py # Streamlit application (detection + analytics)
├── NEXUS.html # Frontend dashboard UI
├── input.jpg # Sample input image
├── annotated.jpg # Sample detection output
├── README.md


---

## How to Run
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install streamlit ultralytics opencv-python numpy pandas pillow folium plotly
Place the YOLO .pt model file in the project root

Run the application:
streamlit run app.py
