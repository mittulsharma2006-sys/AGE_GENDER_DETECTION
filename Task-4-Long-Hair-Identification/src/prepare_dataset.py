import os
import pickle
import numpy as np
import zipfile
import shutil
import random

ZIP_PATH = "data.zip"
PICKLE_PATH = "raw_data/dataset_all.pkl"

OUTPUT_DIR = "dataset"

random.seed(42)

# Load UPAR annotations
print("Loading annotations...")
with open(PICKLE_PATH, "rb") as f:
    d = pickle.load(f)

names = np.array(d.image_name)
labels = np.array(d.label)

# Find PA100K images
pa_idx = np.where(
    np.char.startswith(names.astype(str), "PA100k/")
)[0]

short_idx = []
long_idx = []

for i in pa_idx:
    short = labels[i, 4]
    long = labels[i, 5]
    bald = labels[i, 6]

    # Ignore bald and ambiguous short+long samples
    if bald == 1:
        continue

    if short == 1 and long == 0:
        short_idx.append(i)

    elif long == 1 and short == 0:
        long_idx.append(i)

print(f"Available short images: {len(short_idx)}")
print(f"Available long images: {len(long_idx)}")

# Select balanced samples
random.shuffle(short_idx)
random.shuffle(long_idx)

short_idx = short_idx[:5000]
long_idx = long_idx[:5000]

selected = [
    ("short", i) for i in short_idx
] + [
    ("long", i) for i in long_idx
]

random.shuffle(selected)

# Create folders
for split in ["train", "val", "test"]:
    for category in ["short", "long"]:
        os.makedirs(
            os.path.join(OUTPUT_DIR, split, category),
            exist_ok=True
        )

# Open ZIP without extracting everything
print("Opening data.zip...")
with zipfile.ZipFile(ZIP_PATH, "r") as z:

    total = len(selected)

    for count, (category, i) in enumerate(selected, 1):

        # PA100K path from annotation
        annotation_path = names[i]

        # Convert to path inside ZIP
        zip_path = "release_data/release_data/" + os.path.basename(annotation_path)

        # Determine split
        if count <= 7000:
            split = "train"
        elif count <= 8500:
            split = "val"
        else:
            split = "test"

        output_path = os.path.join(
            OUTPUT_DIR,
            split,
            category,
            os.path.basename(annotation_path)
        )

        with z.open(zip_path) as source, open(output_path, "wb") as target:
            shutil.copyfileobj(source, target)

        if count % 500 == 0:
            print(f"Processed {count}/{total}")

print("\nDataset preparation complete!")

for split in ["train", "val", "test"]:
    for category in ["short", "long"]:
        folder = os.path.join(OUTPUT_DIR, split, category)
        count = len(os.listdir(folder))
        print(f"{split}/{category}: {count}")