\# Task 4 — Long Hair Identification



\## Description



This project implements the Long Hair Identification task using a custom-trained machine learning model and a Streamlit GUI.



The system detects:



\- Age

\- Gender

\- Hair length

\- Final gender according to the Task 4 rules



The hair-length model classifies hair as either \*\*Long\*\* or \*\*Short\*\*.



\## Task 4 Logic



The special hair-based gender logic is applied only when the detected age is between \*\*20 and 30 years inclusive\*\*.



| Age | Hair | Final Gender |

|---|---|---|

| 20–30 | Long | Female |

| 20–30 | Short | Male |

| Below 20 | Any | Detected gender |

| Above 30 | Any | Detected gender |



Therefore, for people aged 20–30, hair length overrides the detected gender.



For people outside this age range, the detected gender is retained regardless of hair length.



\## Machine Learning Model



A MobileNetV2-based transfer-learning model was trained specifically for hair-length classification.



\### Dataset



The project uses:



\- \*\*PA-100K\*\* pedestrian images

\- \*\*UPAR\*\* unified pedestrian attribute annotations



The UPAR annotations provide hair-length labels including:



\- Hair-Length-Short

\- Hair-Length-Long

\- Hair-Length-Bald



Bald and ambiguous Short+Long samples were excluded.



A balanced dataset of 10,000 images was prepared:



\- 5,000 Short Hair

\- 5,000 Long Hair



\### Dataset Split



| Split | Short | Long | Total |

|---|---:|---:|---:|

| Training | 3,513 | 3,487 | 7,000 |

| Validation | 732 | 768 | 1,500 |

| Testing | 755 | 745 | 1,500 |



\## Model Performance



The trained hair-length classifier achieved:



\*\*Test Accuracy: 74.13%\*\*



Classification performance:



| Class | Precision | Recall | F1-Score |

|---|---:|---:|---:|

| Long | 0.75 | 0.72 | 0.73 |

| Short | 0.73 | 0.76 | 0.75 |



\## Technologies Used



\- Python

\- TensorFlow

\- Keras

\- MobileNetV2

\- InsightFace

\- OpenCV

\- Streamlit

\- NumPy

\- Pillow

\- Scikit-learn



\## Project Structure



```text

Task-4-Long-Hair-Identification/

│

├── app.py

├── README.md

├── .gitignore

│

├── models/

│   └── hair\_length\_model.keras

│

├── src/

│   ├── prepare\_dataset.py

│   ├── train\_hair\_model.py

│   └── evaluate\_model.py

│

├── screenshots/

│

├── dataset/

├── raw\_data/

├── data.zip

└── venv/

