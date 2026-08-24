# 🤟 Sign Language Detection

An AI-powered Sign Language Detection application that recognizes American Sign Language (ASL) hand signs using a trained deep learning model, MediaPipe hand landmarks, OpenCV, and Streamlit.

## 🚀 Features

- Real-time sign language detection using a webcam
- Image upload for sign prediction
- Recognizes ASL signs from **A to F**
- MediaPipe-based hand landmark detection
- Displays prediction confidence
- Draws hand landmarks on the camera feed
- User-friendly Streamlit interface
- Application operates between **6:00 PM and 10:00 PM**

## 🧠 How It Works

The application follows this pipeline:

**Input Image / Webcam → MediaPipe Hand Detection → 21 Hand Landmarks → Neural Network → Sign Prediction**

MediaPipe extracts the hand's 21 landmarks. Each landmark contains X, Y, and Z coordinates, producing 63 values that are passed to the trained neural network.

The model predicts one of the six supported signs:

**A, B, C, D, E, F**

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- MediaPipe
- OpenCV
- NumPy
- Streamlit
- Streamlit-WebRTC
- Pillow
- PyAV

## 📁 Project Structure

```text
Task-2-Sign-Language-Detection/
│
├── dataset/
│   ├── A/
│   ├── B/
│   ├── C/
│   ├── D/
│   ├── E/
│   └── F/
│
├── models/
│   └── sign_language_model.keras
│
├── raw_dataset/
│
├── sample_images/
│
├── screenshots/
│
├── app.py
├── predict.py
├── train.py
├── requirements.txt
└── README.md