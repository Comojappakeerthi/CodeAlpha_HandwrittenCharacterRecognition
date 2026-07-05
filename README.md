# CodeAlpha — Handwritten Character Recognition (Digits + Letters)

A full-stack web app that recognizes handwritten **digits (0–9)** and
**letters (a–z)**, built for the CodeAlpha Machine Learning Internship,
Task 3: Handwritten Character Recognition.

**Live demo:** _add your deployed Render URL here after deploying_

## Task requirements → what's implemented

| Requirement | Status |
|---|---|
| Identify handwritten characters or alphabets | ✅ Digits AND letters |
| Image processing + deep learning | ✅ Canvas → crop/center/resize → CNN |
| Dataset: MNIST (digits) | ✅ `train_model.py` |
| Dataset: EMNIST (characters) | ✅ `train_emnist_model.py` (EMNIST "letters" split) |
| Model: CNN | ✅ Same architecture, trained separately per dataset |
| Extendable to full word/sentence recognition (CRNN) | 📝 Documented below as the natural next step — not implemented in this submission, since it requires a different architecture (CNN+RNN+CTC) and a line-level handwriting dataset (e.g. IAM), which is a separate project in scope |

## Project structure

```
CodeAlpha_HandwrittenCharacterRecognition/
├── app.py                    # Flask server, /predict API, mode switching
├── train_model.py            # Trains the MNIST digits CNN
├── train_emnist_model.py     # Trains the EMNIST letters CNN
├── requirements.txt          # Runtime deps (for running/deploying the app)
├── requirements-train.txt    # Extra deps needed only to train models
├── Procfile                  # For Render/Heroku-style deployment
├── render.yaml                # Render deployment config
├── .gitignore
├── models/
│   ├── handwritten_model.keras       # created by train_model.py
│   └── emnist_letters_model.keras    # created by train_emnist_model.py
├── images/                    # accuracy/loss plots (created after training)
├── templates/
│   └── index.html
└── static/
    ├── style.css
    └── script.js
```

## How it works

- **Frontend**: an HTML5 `<canvas>` with a Digits/Letters mode toggle. You
  draw a character, and on "Grade it" the canvas is exported as base64 PNG
  and POSTed to `/predict` along with the selected mode.
- **Backend**: Flask loads the right model for the mode (lazily, once),
  preprocesses the image — grayscale, invert, **crop to the drawn strokes'
  bounding box, center on a padded square, resize to 28×28** (this centering
  step matters a lot for real-world accuracy, since MNIST/EMNIST samples are
  always centered) — then runs `model.predict` and returns the label,
  confidence, and full probability distribution.
- Digits use a 10-class softmax (0–9). Letters use a 26-class softmax
  (a–z, case-merged, matching EMNIST's "letters" split).

## Local setup

### 1. Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### 2. Install training dependencies (one-time, to train both models)

```bash
pip install -r requirements-train.txt
```

### 3. Train the digits model

```bash
python train_model.py
```

Downloads MNIST, trains ~a few minutes on CPU, saves
`models/handwritten_model.keras` (~99% test accuracy).

### 4. Train the letters model

```bash
python train_emnist_model.py
```

Downloads EMNIST via `tensorflow-datasets` (larger download, first run
only), trains, saves `models/emnist_letters_model.keras`.

> Note: `plt.show()` in both scripts pops up plot windows on some systems.
> The model is already saved by that point — closing the plot windows (or
> Ctrl+C) just ends the script, it won't lose your trained model.

### 5. Run the web app

```bash
python app.py
```

Open **http://127.0.0.1:5001**, pick Digits or Letters, draw, and click
**Grade it**.

(Port 5001 is used instead of 5000 because macOS's AirPlay Receiver also
listens on 5000 and will intercept the request.)

## API reference

`POST /predict`

Request body:
```json
{
  "image": "data:image/png;base64,iVBORw0KGgo...",
  "mode": "digits"
}
```
`mode` is `"digits"` or `"letters"`.

Response body:
```json
{
  "mode": "digits",
  "prediction": "7",
  "confidence": 0.9931,
  "scores": { "0": 0.0001, "1": 0.0002, "...": "...", "9": 0.001 }
}
```

## Deploying to Render (free tier)

1. Push this project to GitHub first (see next section).
2. Go to https://render.com → sign in with GitHub → **New +** → **Web Service**.
3. Select your `CodeAlpha_HandwrittenCharacterRecognition` repo.
4. Render should auto-detect `render.yaml`. If not, set manually:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`
   - **Environment Variable:** `FLASK_DEBUG=false`
5. Click **Create Web Service**. First build takes several minutes
   (installing TensorFlow). Once live, Render gives you a URL like
   `https://codealpha-handwritten-recognition.onrender.com`.
6. Free-tier services spin down after inactivity — the first request after
   idling can take 30–60 seconds to wake up. This is normal.

Trained model files (`models/*.keras`) must be committed to the repo
(they're a few MB each, well within GitHub's limits) since Render's free
tier has no persistent storage to retrain on — the app loads the models
you already trained locally.

## Pushing to GitHub

From inside the project folder (with `venv/` excluded via `.gitignore`):

```bash
git init
git add .
git commit -m "Handwritten character recognition: CNN on MNIST + EMNIST, Flask web app"
git branch -M main
git remote add origin https://github.com/<your-username>/CodeAlpha_HandwrittenCharacterRecognition.git
git push -u origin main
```

Replace `<your-username>` with your GitHub username, and create the empty
repo on GitHub first (without a README, so it doesn't conflict with the
one already here).

If `git push` asks for a password and rejects it, GitHub requires a
**personal access token** instead of your account password — generate one
under GitHub → Settings → Developer settings → Personal access tokens, and
use that in place of your password when prompted.

## Extending further

- **More letter accuracy**: increase epochs, add data augmentation
  (rotation/shear), or use EMNIST "byclass" (62 classes: digits + upper +
  lowercase) instead of the case-merged "letters" split.
- **Full word/sentence recognition**: move to a CRNN — a CNN feature
  extractor feeding a bidirectional LSTM/GRU sequence model, trained with
  CTC loss on line-level handwriting data (e.g. the IAM Handwriting
  Database) instead of isolated single characters. This also requires a
  different frontend: a wide canvas for a full line of writing instead of
  one boxed character.
