\# Task 6 — Senior Citizen Identification



\## Project Overview



This project implements a machine learning and computer vision system for identifying senior citizens in a real-time webcam environment such as a mall, store, or local business.



The system detects people using YOLO11, tracks each person using a unique ID, detects their face, estimates their age and gender, determines whether they are a senior citizen, and records their visit information in a CSV file.



The project is designed to support multiple people appearing simultaneously in the camera feed.



\---



\## Objective



The main objectives of this task are:



\- Detect multiple persons in a video or real-time webcam feed.

\- Track each detected person using a unique ID.

\- Detect the person's face.

\- Predict the person's age.

\- Predict the person's gender.

\- Mark a person as a senior citizen if their age is greater than 60.

\- Record the person's age, gender, senior-citizen status, and time of visit.

\- Store the collected information in a CSV file.



\---



\## Task Logic



The system follows the required logic:



| Condition | Result |

|---|---|

| Age ≤ 60 | Regular person |

| Age > 60 | Senior Citizen |

| Age > 60 | Gender is also displayed |

| New person detected | Visit information is logged |



The senior citizen condition is:



```text

Age > 60

