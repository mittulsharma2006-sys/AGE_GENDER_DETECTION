# Task 4 — Long Hair Identification

## 📌 Project Overview

This project implements the **Long Hair Identification** task using a custom-trained machine learning model and a Streamlit-based graphical user interface (GUI).

The system combines:

- Age detection
- Gender detection
- Hair-length classification
- Conditional gender prediction based on age and hair length

The special hair-based gender logic is applied only to people between **20 and 30 years of age**.

---

## 🎯 Objective

The objective of this task is to build a system that:

1. Detects the person's age.
2. Detects the person's gender.
3. Identifies whether the person has long or short hair.
4. Applies the required Task 4 gender rules.
5. Provides the results through an easy-to-use GUI.

---

## 🧠 Task 4 Logic

The hair-based gender modification is applied only when the detected age is between **20 and 30 years inclusive**.

| Age | Hair Length | Final Gender |
|---|---|---|
| 20–30 | Long | Female |
| 20–30 | Short | Male |
| Below 20 | Any | Detected Gender |
| Above 30 | Any | Detected Gender |

### Logic

For individuals aged **20–30**:

- Long hair → Female
- Short hair → Male

For individuals **below 20 or above 30**:

- The detected gender is retained regardless of hair length.

This implements the exact conditional logic required by the task.

---

## 🤖 Machine Learning Model

### Hair-Length Classification

A custom hair-length classification model was trained using **MobileNetV2 transfer learning**.

The model classifies images into two categories:

- **Long Hair**
- **Short Hair**

The original MobileNetV2 network was used as the feature extractor and a custom classification layer was trained for the hair-length task.

### Age and Gender Detection

**InsightFace** is used during inference to detect:

- Age
- Gender
- Face location

The detected face is used to identify the relevant hair region before passing it to the hair-length classifier.

---

## 📊 Dataset

The project uses the following publicly available datasets and annotations:

### PA-100K

PA-100K is a pedestrian attribute dataset containing **100,000 pedestrian images**.

Official repository:

https://github.com/xh-liu/HydraPlus-Net

Official dataset download:

https://drive.google.com/drive/folders/0B5_Ra3JsEOyOUlhKM0VPZ1ZWR2M?resourcekey=0-CdctEkdX1j2GSMSWWfrPSQ&usp=sharing

### UPAR

UPAR provides unified pedestrian attribute annotations, including hair-length attributes.

Hair-length attributes used:

- Hair-Length-Short
- Hair-Length-Long
- Hair-Length-Bald

Bald samples and ambiguous samples containing both Short and Long labels were excluded.

Official repository:

https://github.com/speckean/upar_dataset

UPAR annotation file:

https://raw.githubusercontent.com/speckean/upar_dataset/main/UPAR/dataset_all.pkl

UPAR Challenge:

https://chalearnlap.cvc.uab.es/dataset/45/description/

---

## 📁 Dataset Preparation

A balanced dataset containing **10,000 images** was prepared from the available PA-100K images and UPAR annotations.

### Dataset Composition

- 5,000 Long Hair images
- 5,000 Short Hair images

Bald and ambiguous samples were excluded from training.

---

## 📈 Dataset Split

| Split | Short Hair | Long Hair | Total |
|---|---:|---:|---:|
| Training | 3,513 | 3,487 | 7,000 |
| Validation | 732 | 768 | 1,500 |
| Testing | 755 | 745 | 1,500 |
| **Total** | **5,000** | **5,000** | **10,000** |

---

## 🏆 Model Performance

The trained hair-length classifier achieved:

### Test Accuracy

**74.13%**

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Long Hair | 0.75 | 0.72 | 0.73 |
| Short Hair | 0.73 | 0.76 | 0.75 |

The model was evaluated on **1,500 previously unseen test images**.

---

## 🛠️ Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- InsightFace
- OpenCV
- Streamlit
- NumPy
- Pillow
- scikit-learn

---

## 📂 Project Structure

```text
Task-4-Long-Hair-Identification/
│
├── app.py
├── README.md
├── .gitignore
│
├── src/
│   ├── prepare_dataset.py
│   ├── train_hair_model.py
│   └── evaluate_model.py
│
├── models/
│   └── hair_length_model.keras
│
├── dataset/
│   ├── train/
│   │   ├── long/
│   │   └── short/
│   │
│   ├── val/
│   │   ├── long/
│   │   └── short/
│   │
│   └── test/
│       ├── long/
│       └── short/
│
├── screenshots/
│
├── raw_data/
├── data.zip
└── venv/