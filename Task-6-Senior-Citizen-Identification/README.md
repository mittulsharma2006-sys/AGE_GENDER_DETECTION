\# Task 6 - Senior Citizen Identification



\## Project Overview



This project implements a computer vision system for identifying senior citizens in a mall, store, or similar environment using a live webcam feed.



The system detects multiple people, tracks them individually, estimates their age and gender, identifies senior citizens based on age, and records their visit information in a CSV file.



\## Objective



The objective of this task is to develop a machine learning and computer vision system that can:



\- Detect multiple people in a video or real-time webcam feed.

\- Track each detected person using a unique ID.

\- Estimate the person's age.

\- Estimate the person's gender.

\- Mark a person as a senior citizen if their age is greater than 60.

\- Record the person's age, gender, senior-citizen status, and visit time.

\- Store the collected information in a CSV file.



\## Task Logic



The system follows this logic:



```text

Webcam / Video

&#x20;     ↓

YOLO Person Detection

&#x20;     ↓

Person Tracking

&#x20;     ↓

Face Detection

&#x20;     ↓

Age \& Gender Prediction

&#x20;     ↓

Is Age > 60?

&#x20;  ↙         ↘

&#x20;Yes          No

&#x20; ↓            ↓

Senior      Regular

Citizen      Person

&#x20; ↓

Record Information

&#x20;     ↓

CSV Log

