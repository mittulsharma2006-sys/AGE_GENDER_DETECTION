import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report


DATA_PATH = "dataset/processed/emotion_features_ml.npz"
MODEL_PATH = "models/emotion_svm_model.pkl"
LABEL_PATH = "models/emotion_svm_labels.npy"


# Load data
data = np.load(DATA_PATH)

X = data["X"]
y = data["y"]

print("Feature shape:", X.shape)
print("Labels shape:", y.shape)


# Encode emotion labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

print("Classes:", encoder.classes_)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# SVM pipeline
model = Pipeline([
    ("scaler", StandardScaler()),

    ("svm", SVC(
        kernel="rbf",
        C=10,
        gamma="scale",
        probability=True,
        random_state=42
    ))
])


# Train
print("\nTraining SVM...")

model.fit(
    X_train,
    y_train
)


# Predict
y_pred = model.predict(X_test)


# Accuracy
accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n==============================")
print("SVM EMOTION MODEL RESULTS")
print("==============================")

print(f"Test Accuracy: {accuracy * 100:.2f}%")


# Classification report
print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=encoder.classes_,
        zero_division=0
    )
)


# Save model
import joblib

os.makedirs("models", exist_ok=True)

joblib.dump(
    model,
    MODEL_PATH
)

np.save(
    LABEL_PATH,
    encoder.classes_
)

print("\nModel saved to:", MODEL_PATH)
print("Labels saved to:", LABEL_PATH)