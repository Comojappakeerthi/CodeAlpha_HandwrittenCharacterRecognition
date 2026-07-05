"""
train_model.py
Trains a CNN on MNIST and saves it as models/handwritten_model.keras
Run this ONCE locally before starting the Flask server (app.py).
"""

import os
import matplotlib
matplotlib.use("Agg")  # so it works without a display (servers, CI, etc.)
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)

# ---------------------------------------------------------
# 1. Load & prepare data
# ---------------------------------------------------------
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print("MNIST Dataset Loaded Successfully!")
print("Training Data Shape:", x_train.shape)
print("Testing Data Shape:", x_test.shape)

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# ---------------------------------------------------------
# 2. Build CNN
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
    layers.Dense(10, activation="softmax"),
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
    x_train,
    y_train,
    epochs=8,
    batch_size=128,
    validation_data=(x_test, y_test),
)

# ---------------------------------------------------------
# 4. Evaluate & save
# ---------------------------------------------------------
loss, accuracy = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

model.save("models/handwritten_model.keras")
print("Model Saved Successfully to models/handwritten_model.keras")

# ---------------------------------------------------------
# 5. Plots
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy")
plt.legend()
plt.grid(True)
plt.savefig("images/accuracy_graph.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Model Loss")
plt.legend()
plt.grid(True)
plt.savefig("images/loss_graph.png")
plt.close()

predictions = model.predict(x_test)
plt.figure(figsize=(12, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title(f"Pred: {predictions[i].argmax()}\nActual: {y_test[i]}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("images/sample_predictions.png")
plt.close()

print("\nSample Predictions Saved!")
print("Training Script Completed Successfully!")
