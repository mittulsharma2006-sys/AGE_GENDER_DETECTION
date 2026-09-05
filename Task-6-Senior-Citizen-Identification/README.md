# Task 6 — Senior Citizen Identification

## 📝 Project Overview

This project implements a real-time machine learning and computer vision system for identifying senior citizens in a webcam or video feed.

The system detects multiple people, assigns a unique tracking ID to each person, detects their face, predicts their age and gender, and identifies whether they are senior citizens.

If the detected age is greater than 60 years, the person is classified as a **Senior Citizen**.

The system also records the person's:

- Age
- Gender
- Senior Citizen status
- Time of visit

The collected information is automatically stored in a CSV file.

---

## 🎯 Objective

The objective of this task is to build a system that:

1. Detects multiple persons in a video or real-time webcam feed.
2. Tracks each detected person using a unique ID.
3. Detects the person's face.
4. Predicts the person's age.
5. Predicts the person's gender.
6. Identifies whether the person is a senior citizen.
7. Records the person's visit information.
8. Stores the results in a CSV file.

---

## 🧠 Task 6 Logic

The senior citizen classification is based on the person's predicted age.

| Age | Gender | Final Result |
|---|---|---|
| ≤ 60 | Any | Regular Person |
| > 60 | Male | Senior Citizen |
| > 60 | Female | Senior Citizen |

### Senior Citizen Rule

A person is classified as a senior citizen when:

```text
Age > 60