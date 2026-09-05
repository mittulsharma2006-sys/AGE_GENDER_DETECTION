import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib


INPUT_FILE = "dataset/processed/age_features.npz"
MODEL_FILE = "models/age_voice_model.pkl"

AGE_LABELS = [
    "teens",
    "twenties",
    "thirties",
    "fourties",
    "fifties",
    "60+"
]


def main():

    os.makedirs("models", exist_ok=True)

    data = np.load(INPUT_FILE)

    X = data["X"]
    y = data["y"]

    print("Dataset shape:", X.shape)
    print("Labels shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            SVC(
                kernel="rbf",
                C=10,
                gamma="scale",
                probability=True
            )
        )
    ])

    print("\nTraining age classifier...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\nAge Classification Accuracy:")
    print(f"{accuracy * 100:.2f}%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            labels=list(range(len(AGE_LABELS))),
            target_names=AGE_LABELS,
            zero_division=0
        )
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print("\nModel saved:")
    print(MODEL_FILE)


if __name__ == "__main__":
    main()