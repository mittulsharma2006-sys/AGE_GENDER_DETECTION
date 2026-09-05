import os
import glob
import numpy as np
import librosa

DATASET_PATH = "dataset/emotion"
OUTPUT_PATH = "dataset/processed/emotion_features_ml.npz"

EMOTIONS = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

X = []
y = []

files = glob.glob(
    os.path.join(DATASET_PATH, "**", "*.wav"),
    recursive=True
)

print(f"Found {len(files)} audio files")

for i, file_path in enumerate(files, start=1):

    filename = os.path.basename(file_path)
    emotion_code = filename.split("-")[2]

    if emotion_code not in EMOTIONS:
        continue

    try:
        audio, sr = librosa.load(
            file_path,
            sr=16000
        )

        # Normalize audio
        audio = librosa.util.normalize(audio)

        # MFCC
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=40,
            n_fft=1024,
            hop_length=512
        )

        # Delta
        delta = librosa.feature.delta(mfcc)

        # Delta-delta
        delta2 = librosa.feature.delta(
            mfcc,
            order=2
        )

        # Statistical representation
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),

            np.mean(delta, axis=1),
            np.std(delta, axis=1),

            np.mean(delta2, axis=1),
            np.std(delta2, axis=1)
        ])

        X.append(features)
        y.append(emotion_code)

    except Exception as e:
        print(f"Error: {file_path}")
        print(e)

    if i % 100 == 0:
        print(f"Processed {i}/{len(files)}")


X = np.array(X, dtype=np.float32)
y = np.array(y)

np.savez(
    OUTPUT_PATH,
    X=X,
    y=y
)

print("\nFeature extraction completed.")
print("Feature shape:", X.shape)
print("Labels shape:", y.shape)
print("Saved to:", OUTPUT_PATH)