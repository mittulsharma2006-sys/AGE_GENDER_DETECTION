import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

TEST_DIR = "dataset/test"
MODEL_PATH = "models/hair_length_model.keras"

# Load test dataset
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="binary",
    shuffle=False
)

print("Classes:", test_ds.class_names)

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Evaluate
loss, accuracy = model.evaluate(test_ds, verbose=1)

print("\nTest Results")
print("-----------")
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Predictions
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images, verbose=0)

    y_true.extend(labels.numpy().flatten().astype(int))
    y_pred.extend((predictions.flatten() >= 0.5).astype(int))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_true,
        y_pred,
        target_names=test_ds.class_names
    )
)