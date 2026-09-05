\# Task 5 — Age and Emotion Detection Through Voice



\## Project Overview



This project implements a machine learning system that analyzes a person's voice recording to detect their gender and approximate age.



The system is designed specifically for male voices. If a female voice is detected, the application rejects the input and displays:



> Upload male voice.



For male voices, the system estimates the person's approximate age.



If the person is classified as being above 60 years old, the application marks them as a \*\*Senior Citizen\*\* and performs emotion detection.



For individuals classified as under 60, only the approximate age is displayed.



The project includes a Streamlit-based graphical user interface (GUI).



\---



\## Objective



The main objectives of this task are:



\- Detect gender from a voice recording.

\- Accept only male voices.

\- Reject female voices with the required message.

\- Predict approximate age from voice.

\- Identify senior citizens.

\- Detect emotion for senior citizens.

\- Build a functional graphical user interface.

\- Create and integrate custom machine learning models.



\---



\## Task Logic



The application follows this logic:



| Condition | Output |

|---|---|

| Female voice | `Upload male voice.` |

| Male voice, age ≤ 60 | Approximate age |

| Male voice, age > 60 | Senior Citizen + Approximate age + Emotion |



The system uses a separate senior-citizen classifier as an additional signal for the 60+ decision.



\---



\## Machine Learning Approach



The project uses audio features extracted from voice recordings.



\### Audio Processing



The uploaded audio is:



1\. Converted to mono.

2\. Resampled to 16 kHz.

3\. Padded or truncated to 4 seconds.

4\. Converted into MFCC-based features.



The feature vector contains:



\- MFCC mean

\- MFCC standard deviation

\- Delta mean

\- Delta standard deviation

\- Delta-delta mean

\- Delta-delta standard deviation



This produces a total of \*\*240 audio features\*\*.



\---



\## Models



\### 1. Gender Detection Model



A Support Vector Machine (SVM) classifier is used to distinguish between male and female voices.



Model:



\- SVM

\- RBF kernel

\- StandardScaler

\- Probability estimation enabled



Test accuracy:



\*\*92.94%\*\*



\---



\### 2. Age Regression Model



An SVM-based regression model is used to estimate approximate age.



The model was trained using representative ages for age groups:



| Age Group | Representative Age |

|---|---:|

| Teens | 16 |

| Twenties | 25 |

| Thirties | 35 |

| Forties | 45 |

| Fifties | 55 |

| Sixties | 65 |

| Seventies | 75 |

| Eighties | 85 |

| Nineties | 95 |



The predicted value is therefore an \*\*approximate age estimate\*\*, rather than an exact age.



Model:



\- StandardScaler

\- RBF SVR

\- C = 10

\- Epsilon = 0.2



Test performance:



\- MAE: \*\*8.95 years\*\*

\- RMSE: \*\*12.51 years\*\*

\- R²: \*\*0.0436\*\*



\---



\### 3. Senior Citizen Classifier



A separate SVM classifier is used as an additional signal for determining whether a male voice belongs to the 60+ age group.



The model uses:



\- Under 60

\- 60+



The training data was balanced between the two classes.



The classifier achieved approximately \*\*59.66% balanced test accuracy\*\*.



Because the age model provides an approximate estimate, the senior classifier is used as a supporting signal when determining the Senior Citizen branch.



\---



\### 4. Emotion Detection Model



Emotion detection is performed only when the person is classified as a senior citizen.



The model uses MFCC-based audio features and an SVM classifier.



The emotion dataset contains the following RAVDESS emotion classes:



| Code | Emotion |

|---|---|

| 01 | Neutral |

| 02 | Calm |

| 03 | Happy |

| 04 | Sad |

| 05 | Angry |

| 06 | Fearful |

| 07 | Disgust |

| 08 | Surprised |



Test accuracy:



\*\*64.58%\*\*



\---



\## Datasets



\### Common Voice



Common Voice data was used for:



\- Gender classification

\- Age-related voice modelling

\- Senior citizen classification



The selected metadata contains self-reported age and gender information.



\### RAVDESS



The RAVDESS speech dataset was used for emotion detection.



The 16 kHz speech version was used for model development.



\---



\## Dataset Preparation



For gender and age-related models, male and female voice samples were selected from the Common Voice metadata.



The gender dataset was balanced:



\- 8,000 male samples

\- 8,000 female samples

\- Total: 16,000 samples



For age modelling, male voice samples were used.



For senior classification, the data was divided into:



\- Under 60

\- 60+



The emotion model was trained using RAVDESS speech recordings.



\---



\## Technologies Used



\- Python

\- Streamlit

\- Scikit-learn

\- Librosa

\- SoundFile

\- NumPy

\- Pandas

\- Joblib

\- Matplotlib

\- RAVDESS

\- Common Voice



\---



\## Project Structure



```text

Task-5-Age-Emotion-Voice/

│

├── app.py

├── requirements.txt

├── .gitignore

├── README.md

│

├── models/

│   ├── gender\_voice\_model.pkl

│   ├── age\_regression\_model.pkl

│   ├── senior\_voice\_model.pkl

│   ├── emotion\_svm\_model.pkl

│   └── emotion\_svm\_labels.npy

│

├── src/

│   ├── extract\_gender\_features.py

│   ├── train\_gender\_model.py

│   ├── extract\_age\_features.py

│   ├── train\_age\_model.py

│   ├── extract\_age\_regression\_features.py

│   ├── train\_age\_regression\_model.py

│   ├── extract\_senior\_features.py

│   ├── train\_senior\_model.py

│   ├── train\_emotion\_model.py

│   └── get\_senior\_test\_audio.py

│

├── screenshots/

│

├── dataset/

│

└── sample\_audio/

