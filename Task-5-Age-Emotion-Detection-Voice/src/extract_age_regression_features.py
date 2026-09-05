import os
import io
import pickle
import numpy as np
import librosa


INPUT_FILE = "dataset/gender_age_metadata.pkl"
OUTPUT_FILE = "dataset/processed/age_regression_features.npz"

SAMPLE_RATE = 16000
DURATION = 4
MAX_SAMPLES = SAMPLE_RATE * DURATION


# Convert Common Voice age groups
# into representative approximate ages.
AGE_MAP = {
    "teens": 16,
    "twenties": 25,
    "thirties": 35,
    "fourties": 45,
    "fifties": 55,
    "sixties": 65,
    "seventies": 75,
    "eighties": 85,
    "nineties": 95,
}


def extract_features(audio_bytes):

    audio, sr = librosa.load(
        io.BytesIO(audio_bytes),
        sr=SAMPLE_RATE,
        mono=True
    )

    if len(audio) < MAX_SAMPLES:
        audio = np.pad(
            audio,
            (0, MAX_SAMPLES - len(audio))
        )
    else:
        audio = audio[:MAX_SAMPLES]

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=40
    )

    delta = librosa.feature.delta(mfcc)

    delta2 = librosa.feature.delta(
        mfcc,
        order=2
    )

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

    print("\nExtracting age regression features...")

    for index, row in df.iterrows():

        # Age model is trained only on male voices
        if row["gender"] != "male_masculine":
            continue

        age_group = row["age"]

        if age_group not in AGE_MAP:
            continue

        try:

            audio = row["audio"]
            audio_bytes = audio["bytes"]

            features = extract_features(
                audio_bytes
            )

            X.append(features)
            y.append(AGE_MAP[age_group])

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
    print("Age labels shape:", y.shape)

    print("\nApproximate age distribution:")

    unique, counts = np.unique(
        y,
        return_counts=True
    )

    for age, count in zip(unique, counts):
        print(
            f"{age}: {count}"
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