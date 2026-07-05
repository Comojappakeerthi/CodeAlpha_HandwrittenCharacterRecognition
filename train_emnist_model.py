

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers

os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)

# ---------------------------------------------------------
# 1. Load EMNIST letters split (labels 1-26 -> a-z, case-merged)
# ---------------------------------------------------------
print("Loading EMNIST 'letters' dataset (downloads on first run)...")
(ds_train, ds_test), ds_info = tfds.load(
    "emnist/letters",
    split=["train", "test"],
    as_supervised=True,
    with_info=True,
)

print("Train examples:", ds_info.splits["train"].num_examples)
print("Test examples:", ds_info.splits["test"].num_examples)


def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0

    # EMNIST source images are stored rotated/mirrored relative to the
    # natural reading orientation; this fixes that.
    image = tf.image.rot90(image, k=3)
    image = tf.image.flip_left_right(image)

    # Original labels are 1-26 -> remap to 0-25
    label = label - 1

    return image, label


BATCH_SIZE = 128

ds_train = (
    ds_train.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(10_000)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)
ds_test = (
    ds_test.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

# ---------------------------------------------------------
# 2. Build CNN (same architecture as digits, 26-way output)
# ---------------------------------------------------------
model = keras.Sequential([
    layers.Input(shape=(28, 28, 1)),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(128, activation="relu"),
    layers.Dense(26, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ---------------------------------------------------------
# 3. Train
# ---------------------------------------------------------
print("\nTraining Started...\n")

history = model.fit(
    ds_train,
    epochs=8,
    validation_data=ds_test,
)

# ---------------------------------------------------------
# 4. Evaluate & save
# ---------------------------------------------------------
loss, accuracy = model.evaluate(ds_test)
print(f"\nEMNIST Letters Test Accuracy: {accuracy * 100:.2f}%")

model.save("models/emnist_letters_model.keras")
print("Model Saved Successfully to models/emnist_letters_model.keras")

# ---------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("EMNIST Letters — Model Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("images/emnist_accuracy_graph.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("EMNIST Letters — Model Loss")
plt.legend()
plt.grid(True)
plt.savefig("images/emnist_loss_graph.png")
plt.close()

print("\nEMNIST training script completed successfully!")
