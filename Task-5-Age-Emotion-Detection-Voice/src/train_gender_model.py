import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib


INPUT_FILE = "dataset/processed/gender_features.npz"
MODEL_FILE = "models/gender_voice_model.pkl"


def main():

    os.makedirs("models", exist_ok=True)

    data = np.load(INPUT_FILE)

    X = data["X"]
    y = data["y"]

    print("Dataset shape:", X.shape)
    print("Labels shape:", y.shape)

    # 80% training / 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Standardization + RBF SVM
    model = Pipeline([
        (
            "scaler",
            StandardScaler()
        ),
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

    print("\nTraining gender classifier...")

    model.fit(
        X_train,
        y_train
    )

    # Prediction
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\nGender Classification Accuracy:")
    print(f"{accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            target_names=[
                "Female",
                "Male"
            ]
        )
    )

    # Save model
    joblib.dump(
        model,
        MODEL_FILE
    )

    print("\nModel saved:")
    print(MODEL_FILE)


if __name__ == "__main__":
    main()