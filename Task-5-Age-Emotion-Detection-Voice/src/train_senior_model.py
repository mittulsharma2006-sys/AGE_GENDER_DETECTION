import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix
)


INPUT_FILE = "dataset/processed/senior_features.npz"
MODEL_FILE = "models/senior_voice_model.pkl"


def main():
    os.makedirs("models", exist_ok=True)

    data = np.load(INPUT_FILE)

    X = data["X"]
    y = data["y"]

    print("Original dataset:")
    print("Under 60:", np.sum(y == 0))
    print("60+:", np.sum(y == 1))

    # Separate classes
    X_under60 = X[y == 0]
    X_senior = X[y == 1]

    # Balance the dataset
    rng = np.random.RandomState(42)

    selected_indices = rng.choice(
        len(X_under60),
        size=len(X_senior),
        replace=False
    )

    X_under60 = X_under60[selected_indices]

    # Combine balanced classes
    X_balanced = np.vstack([
        X_under60,
        X_senior
    ])

    y_balanced = np.concatenate([
        np.zeros(len(X_under60)),
        np.ones(len(X_senior))
    ])

    # Shuffle
    shuffle_indices = rng.permutation(len(X_balanced))

    X_balanced = X_balanced[shuffle_indices]
    y_balanced = y_balanced[shuffle_indices]

    print("\nBalanced dataset:")
    print("Under 60:", np.sum(y_balanced == 0))
    print("60+:", np.sum(y_balanced == 1))
    print("Total:", len(y_balanced))

    X_train, X_test, y_train, y_test = train_test_split(
        X_balanced,
        y_balanced,
        test_size=0.20,
        random_state=42,
        stratify=y_balanced
    )

    print("\nTraining samples:", len(X_train))
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

    print("\nTraining balanced senior citizen classifier...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nSenior Citizen Classification Results:")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            target_names=["Under 60", "60+"],
            digits=2
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(model, MODEL_FILE)

    print("\nModel saved:")
    print(MODEL_FILE)


if __name__ == "__main__":
    main()