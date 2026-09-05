import os
import pandas as pd

INPUT_FILE = "dataset/gender_age_metadata.pkl"
OUTPUT_DIR = "sample_audio"

os.makedirs(OUTPUT_DIR, exist_ok=True)

df = pd.read_pickle(INPUT_FILE)

# Find male speakers aged 60+
senior_df = df[
    (df["gender"] == "male_masculine") &
    (df["age"].isin([
        "sixties",
        "seventies",
        "eighties",
        "nineties"
    ]))
]

print("Senior male samples found:", len(senior_df))

if len(senior_df) == 0:
    print("No senior male sample found.")
    exit()

row = senior_df.iloc[0]

audio_bytes = row["audio"]["bytes"]

output_file = os.path.join(
    OUTPUT_DIR,
    "male_60plus_test.wav"
)

with open(output_file, "wb") as f:
    f.write(audio_bytes)

print("\nAge group:", row["age"])
print("Gender:", row["gender"])
print("Saved:")
print(output_file)