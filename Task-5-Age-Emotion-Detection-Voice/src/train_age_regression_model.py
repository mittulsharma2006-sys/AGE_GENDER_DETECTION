import os
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


INPUT_FILE = "dataset/processed/age_regression_features.npz"
MODEL_FILE = "models/age_regression_model.pkl"


def main():

    os.makedirs("models", exist_ok=True)

    data = np.load(INPUT_FILE)

    X = data["X"]
    y = data["y"]

    print("Dataset shape:", X.shape)
    print("Age labels shape:", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print("Training samples:", len(X_train))
    print("Testing samples:", len(X_test))

    model = Pipeline([
        ("scaler", StandardScaler()),
        (
            "regressor",
            SVR(
                kernel="rbf",
                C=10,
                gamma="scale",
                epsilon=0.2
            )
        )
    ])

    print("\nTraining age regression model...")

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            y_pred
        )
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    print("\nAge Regression Results:")
    print(f"MAE:  {mae:.2f} years")
    print(f"RMSE: {rmse:.2f} years")
    print(f"R²:   {r2:.4f}")

    joblib.dump(
        model,
        MODEL_FILE
    )

    print("\nModel saved:")
    print(MODEL_FILE)


if __name__ == "__main__":
    main()