"""
app.py
Flask backend for the Handwritten Character Recognition web app.

Supports two modes:
    - "digits": CNN trained on MNIST (0-9)
    - "letters": CNN trained on EMNIST letters (a-z, case-merged)

Endpoints:
    GET  /              -> serves the frontend (templates/index.html)
    POST /predict        -> accepts a base64 PNG (from the canvas) and a
                             "mode" field, returns the predicted character
                             + confidence scores for that mode

Run locally:
    python app.py
Then open http://127.0.0.1:5001 in your browser.

Make sure you've already run train_model.py (digits) and, optionally,
train_emnist_model.py (letters) so the model files exist under models/.
"""

import base64
import io
import os
import string

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageOps
from tensorflow import keras

app = Flask(__name__)

MODEL_PATHS = {
    "digits": os.path.join("models", "handwritten_model.keras"),
    "letters": os.path.join("models", "emnist_letters_model.keras"),
}

NUM_CLASSES = {
    "digits": 10,
    "letters": 26,
}

_loaded_models = {}


def load_model(mode: str):
    if mode not in MODEL_PATHS:
        raise ValueError(f"Unknown mode '{mode}'. Expected 'digits' or 'letters'.")

    if mode not in _loaded_models:
        path = MODEL_PATHS[mode]
        if not os.path.exists(path):
            training_script = (
                "train_model.py" if mode == "digits" else "train_emnist_model.py"
            )
            raise FileNotFoundError(
                f"Model not found at {path}. Run `python {training_script}` "
                "first to train and save it."
            )
        _loaded_models[mode] = keras.models.load_model(path)

    return _loaded_models[mode]


def label_for(mode: str, class_index: int) -> str:
    if mode == "digits":
        return str(class_index)
    return string.ascii_lowercase[class_index]


def preprocess_image(image_data_url: str) -> np.ndarray:
    """
    Convert a base64 data URL (from the HTML canvas) into a
    28x28x1 normalized numpy array ready for the model.

    Steps:
      1. Decode + grayscale
      2. Invert (canvas is dark-on-light; MNIST/EMNIST are light-on-dark)
      3. Crop to the drawn strokes' bounding box
      4. Pad to a square with margin, then resize to 28x28
         (centers the character the way MNIST/EMNIST samples are centered,
         which meaningfully improves real-world accuracy)
    """
    if "," in image_data_url:
        image_data_url = image_data_url.split(",", 1)[1]

    image_bytes = base64.b64decode(image_data_url)
    image = Image.open(io.BytesIO(image_bytes)).convert("L")
    image = ImageOps.invert(image)

    arr = np.array(image)
    threshold = 20
    ys, xs = np.where(arr > threshold)

    if len(xs) > 0 and len(ys) > 0:
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        cropped = image.crop((x_min, y_min, x_max + 1, y_max + 1))

        w, h = cropped.size
        side = int(max(w, h) * 1.4)
        square = Image.new("L", (side, side), color=0)
        paste_x = (side - w) // 2
        paste_y = (side - h) // 2
        square.paste(cropped, (paste_x, paste_y))
        image = square

    image = image.resize((28, 28), Image.LANCZOS)

    arr = np.array(image).astype("float32") / 255.0
    arr = arr.reshape(1, 28, 28, 1)
    return arr


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        image_data_url = data.get("image")
        mode = data.get("mode", "digits")

        if not image_data_url:
            return jsonify({"error": "No image provided"}), 400
        if mode not in MODEL_PATHS:
            return jsonify({"error": f"Invalid mode '{mode}'"}), 400

        arr = preprocess_image(image_data_url)
        clf = load_model(mode)
        predictions = clf.predict(arr, verbose=0)[0]

        predicted_index = int(np.argmax(predictions))
        confidence = float(np.max(predictions))
        n = NUM_CLASSES[mode]
        all_scores = {label_for(mode, i): float(predictions[i]) for i in range(n)}

        return jsonify(
            {
                "mode": mode,
                "prediction": label_for(mode, predicted_index),
                "confidence": confidence,
                "scores": all_scores,
            }
        )
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    debug_mode = os.environ.get("FLASK_DEBUG", "true").lower() == "true"
    app.run(debug=debug_mode, host="0.0.0.0", port=port)
