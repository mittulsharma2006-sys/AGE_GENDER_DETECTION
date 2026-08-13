# AI Detection System - Age, Gender & Traffic Analysis

## 📌 Project Overview

This project was originally developed as an **Age and Gender Detection system using Deep Learning** during the training phase.

As an extension of the same project, a **Traffic Analysis module** was implemented as an additional internship feature. The extended application can detect and count cars and people in traffic images and identify blue cars based on their colour.

The complete application is integrated into a single **Streamlit GUI**.

---

## 🚀 Features

### 👤 Original Training Project - Age & Gender Detection

- Age prediction from images
- Gender prediction
- Prediction confidence
- Image upload through Streamlit
- Deep learning model using TensorFlow/Keras
- Image preprocessing

### 🚦 Internship Extension - Traffic Analysis

- Detect cars in traffic images
- Count the total number of cars
- Detect blue cars
- Count other-colour cars
- Detect people
- Count people at the traffic signal
- Display bounding boxes around detected objects
- **Red rectangle for blue cars**
- **Blue rectangle for other-colour cars**
- **Green rectangle for people**
- Preview uploaded traffic images
- Display detection statistics
- Integrated into the existing project GUI

---

## 🖥️ Application Interface

The application provides two options through the sidebar:

1. **Age & Gender Detection**
2. **Traffic Analysis**

This allows the internship functionality to be used as an extension of the original training project rather than as a separate application.

---

## 🧠 Technologies Used

- Python
- TensorFlow
- Keras
- YOLO / Ultralytics
- OpenCV
- NumPy
- Pillow
- Matplotlib
- Streamlit

---

## 📂 Project Structure

```text
AGE_GENDER_DETECTION/
│
├── app.py
├── Age_Sex_Detection.h5
├── yolo11n.pt
├── requirements.txt
├── README.md
│
├── modules/
│   ├── __init__.py
│   ├── colour_detection.py
│   └── traffic_detection.py
│
├── sample_images/
│   └── traffic.jpg
│
├── test_traffic.py
│
└── age-gender-identification.ipynb