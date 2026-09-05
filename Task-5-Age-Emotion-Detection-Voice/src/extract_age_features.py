import os
import io
import pickle
import numpy as np
import pandas as pd
import librosa


INPUT_FILE = "dataset/gender_age_metadata.pkl"
OUTPUT_FILE = "dataset/processed/age_features.npz"

SAMPLE_RATE = 16000
DURATION = 4
MAX_SAMPLES = SAMPLE_RATE * DURATION


# Age classes
# All ages 60 and above are combined into one class
AGE_MAP = {
    "teens": 0,
    "twenties": 1,
    "thirties": 2,
    "fourties": 3,
    "fifties": 4,
    "sixties": 5,
    "seventies": 5,
    "eighties": 5,
    "nineties": 5,
}

AGE_LABELS = [
    "teens",
    "twenties",
    "thirties",
    "fourties",
    "fifties",
    "60+",
]


def extract_features(audio_bytes):

    audio, sr = librosa.load(
        io.BytesIO(audio_bytes),
        sr=SAMPLE_RATE,
        mono=True
    )

    # Pad or truncate to 4 seconds
    if len(audio) < MAX_SAMPLES:
        audio = np.pad(
            audio,
            (0, MAX_SAMPLES - len(audio))
        )
    else:
        audio = audio[:MAX_SAMPLES]

    # MFCC
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    # First derivative
    delta = librosa.feature.delta(mfcc)

    # Second derivative
    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

    # Same 240-feature representation
    features = np.concatenate([
        mfcc.mean(axis=1),
        mfcc.std(axis=1),

        delta.mean(axis=1),
        delta.std(axis=1),

        delta2.mean(axis=1),
        delta2.std(axis=1)
    ])

    return features


def main():

    os.makedirs(
        "dataset/processed",
        exist_ok=True
    )

    print("Loading metadata...")

    with open(INPUT_FILE, "rb") as f:
        df = pickle.load(f)

    print("Metadata type:", type(df))
    print("Total samples:", len(df))

    X = []
    y = []

    print("\nExtracting age features...")

    for index, row in df.iterrows():

        gender = row["gender"]
        age = row["age"]

        # Only male voices
        if gender != "male_masculine":
            continue

        # Skip unknown age values
        if age not in AGE_MAP:
            continue

        try:

            audio = row["audio"]

            # Audio is stored as a dictionary
            audio_bytes = audio["bytes"]

            features = extract_features(
                audio_bytes
            )

            X.append(features)
            y.append(AGE_MAP[age])

        except Exception as e:

            print(
                f"Skipping sample {index}: {e}"
            )

        if len(X) % 500 == 0:
            print(
                f"Processed audio samples: {len(X)}"
            )

    X = np.array(X)
    y = np.array(y)

    print("\nFeature shape:", X.shape)
    print("Labels shape:", y.shape)

    print("\nAge distribution:")

    for label_id, label_name in enumerate(AGE_LABELS):

        count = np.sum(
            y == label_id
        )

        print(
            f"{label_name}: {count}"
        )

    np.savez(
        OUTPUT_FILE,
        X=X,
        y=y
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()