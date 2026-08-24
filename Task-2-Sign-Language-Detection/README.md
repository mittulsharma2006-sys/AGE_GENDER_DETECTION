# Sign Language Detection — Task 2

## Overview

This project is a machine learning based Sign Language Detection system developed as part of Internship Task 2.

The system can recognize selected American Sign Language (ASL) signs using hand landmarks extracted with MediaPipe and a trained neural network model.

The application provides a Streamlit GUI with:

- Upload Image detection
- Real-Time Camera detection
- Predicted sign
- Confidence score
- Hand landmark visualization
- Time-based operation from 6 PM to 10 PM

## Supported Signs

The current model supports:

**A, B, C, D, E and F**

## Technologies Used

- Python
- TensorFlow
- Keras
- MediaPipe
- OpenCV
- NumPy
- Streamlit
- Streamlit WebRTC
- scikit-learn

## Project Structure

```text
Task-2-Sign-Language-Detection/
│
├── models/
│   └── sign_language_model.keras
│
├── sample_images/
│
├── screenshots/
│   ├── upload_image.png
│   ├── real_time.png
│   └── time_restriction.png
│
├── app.py
├── predict.py
├── train.py
├── requirements.txt
└── README.md