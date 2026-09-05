import os
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from io import BytesIO
from tqdm import tqdm

INPUT_FILE = "dataset/gender_age_metadata.pkl"
OUTPUT_FILE = "dataset/processed/gender_features.npz"

SR = 16000
DURATION = 4
N_MFCC = 40


def extract_features(audio_data):
    try:
        # Decode audio bytes
        audio_bytes = audio_data["bytes"]

        y, sr = sf.read(
            BytesIO(audio_bytes),
            dtype="float32"
        )

        # Convert stereo to mono
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        # Resample if necessary
        if sr != SR:
            y = librosa.resample(
                y,
                orig_sr=sr,
                target_sr=SR
            )

        target_length = SR * DURATION

        # Pad short audio
        if len(y) < target_length:
            y = np.pad(
                y,
                (0, target_length - len(y))
            )
        else:
            y = y[:target_length]

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=y,
            sr=SR,
            n_mfcc=N_MFCC
        )

        # Delta
        delta = librosa.feature.delta(mfcc)

        # Delta-delta
        delta2 = librosa.feature.delta(
            mfcc,
            order=2
        )

        # Fixed-size feature vector
        features = np.concatenate([
            mfcc.mean(axis=1),
            mfcc.std(axis=1),

            delta.mean(axis=1),
            delta.std(axis=1),

            delta2.mean(axis=1),
            delta2.std(axis=1)
        ])

        return features

    except Exception as e:
        print("Audio processing error:", e)
        return None


def main():

    os.makedirs(
        "dataset/processed",
        exist_ok=True
    )

    df = pd.read_pickle(INPUT_FILE)

    features = []
    labels = []

    print("Extracting gender features...")
    print("Total samples:", len(df))

    for _, row in tqdm(
        df.iterrows(),
        total=len(df)
    ):

        feature = extract_features(
            row["audio"]
        )

        if feature is not None:

            features.append(feature)

            # Male = 1
            # Female = 0
            label = (
                1
                if row["gender"] == "male_masculine"
                else 0
            )

            labels.append(label)

    X = np.array(
        features,
        dtype=np.float32
    )

    y = np.array(
        labels,
        dtype=np.int64
    )

    np.savez_compressed(
        OUTPUT_FILE,
        X=X,
        y=y
    )

    print("\nFeature extraction complete!")
    print("Feature shape:", X.shape)
    print("Labels shape:", y.shape)
    print("Male:", np.sum(y == 1))
    print("Female:", np.sum(y == 0))
    print("Saved:", OUTPUT_FILE)


if __name__ == "__main__":
    main()