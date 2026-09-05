import os
import io
import numpy as np
import pandas as pd
import librosa
import soundfile as sf

INPUT_FILE = "dataset/gender_age_metadata.pkl"
OUTPUT_FILE = "dataset/processed/senior_features.npz"

AGE_MAP = {
    "teens": 0,
    "twenties": 0,
    "thirties": 0,
    "fourties": 0,
    "fifties": 0,
    "sixties": 1,
    "seventies": 1,
    "eighties": 1,
    "nineties": 1,
}


def extract_features(audio_bytes):
    audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")

    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    if sr != 16000:
        audio = librosa.resample(
            audio,
            orig_sr=sr,
            target_sr=16000
        )

    target_length = 16000 * 4

    if len(audio) < target_length:
        audio = np.pad(
            audio,
            (0, target_length - len(audio))
        )
    else:
        audio = audio[:target_length]

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=16000,
        n_mfcc=40
    )

    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = np.concatenate([
        np.mean(mfcc, axis=1),
        np.std(mfcc, axis=1),
        np.mean(delta, axis=1),
        np.std(delta, axis=1),
        np.mean(delta2, axis=1),
        np.std(delta2, axis=1)
    ])

    return features


def main():
    os.makedirs("dataset/processed", exist_ok=True)

    df = pd.read_pickle(INPUT_FILE)

    df = df[
        (df["gender"] == "male_masculine") &
        (df["age"].isin(AGE_MAP.keys()))
    ].copy()

    X = []
    y = []

    print("Total male samples:", len(df))

    for i, (_, row) in enumerate(df.iterrows()):
        try:
            audio_bytes = row["audio"]["bytes"]

            features = extract_features(audio_bytes)

            X.append(features)
            y.append(AGE_MAP[row["age"]])

        except Exception as e:
            print("Skipping sample:", i, e)

        if (i + 1) % 500 == 0:
            print("Processed:", i + 1)

    X = np.array(X)
    y = np.array(y)

    print("\nFeature shape:", X.shape)
    print("Labels shape:", y.shape)

    print("\nUnder 60:", np.sum(y == 0))
    print("60+:", np.sum(y == 1))

    np.savez(
        OUTPUT_FILE,
        X=X,
        y=y
    )

    print("\nSaved:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()