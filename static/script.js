const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const clearBtn = document.getElementById("clearBtn");
const predictBtn = document.getElementById("predictBtn");
const modeDigitsBtn = document.getElementById("modeDigits");
const modeLettersBtn = document.getElementById("modeLetters");
const hintText = document.getElementById("hintText");

let currentMode = "digits";

function setMode(mode) {
  currentMode = mode;
  const isDigits = mode === "digits";

  modeDigitsBtn.classList.toggle("active", isDigits);
  modeDigitsBtn.setAttribute("aria-selected", String(isDigits));
  modeLettersBtn.classList.toggle("active", !isDigits);
  modeLettersBtn.setAttribute("aria-selected", String(!isDigits));

  hintText.textContent = isDigits
    ? "Draw one digit, 0–9, filling most of the box."
    : "Draw one letter, a–z, filling most of the box.";

  initCanvas();
  resultFilled.hidden = true;
  resultError.hidden = true;
  resultEmpty.hidden = false;
}

modeDigitsBtn.addEventListener("click", () => setMode("digits"));
modeLettersBtn.addEventListener("click", () => setMode("letters"));

const resultEmpty = document.getElementById("resultEmpty");
const resultFilled = document.getElementById("resultFilled");
const resultError = document.getElementById("resultError");
const errorText = document.getElementById("errorText");
const predictedDigitEl = document.getElementById("predictedDigit");
const confidenceValueEl = document.getElementById("confidenceValue");
const scoresEl = document.getElementById("scores");

let drawing = false;
let lastX = 0;
let lastY = 0;

function initCanvas() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 16;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#2b2b2e";
}
initCanvas();

function getPos(evt) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  if (evt.touches && evt.touches.length > 0) {
    return {
      x: (evt.touches[0].clientX - rect.left) * scaleX,
      y: (evt.touches[0].clientY - rect.top) * scaleY,
    };
  }
  return {
    x: (evt.clientX - rect.left) * scaleX,
    y: (evt.clientY - rect.top) * scaleY,
  };
}

function startDraw(evt) {
  evt.preventDefault();
  drawing = true;
  const pos = getPos(evt);
  lastX = pos.x;
  lastY = pos.y;
}

function draw(evt) {
  if (!drawing) return;
  evt.preventDefault();
  const pos = getPos(evt);
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
  lastX = pos.x;
  lastY = pos.y;
}

function endDraw() {
  drawing = false;
}

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", draw);
canvas.addEventListener("mouseup", endDraw);
canvas.addEventListener("mouseleave", endDraw);

canvas.addEventListener("touchstart", startDraw, { passive: false });
canvas.addEventListener("touchmove", draw, { passive: false });
canvas.addEventListener("touchend", endDraw);

clearBtn.addEventListener("click", () => {
  initCanvas();
  resultFilled.hidden = true;
  resultError.hidden = true;
  resultEmpty.hidden = false;
});

predictBtn.addEventListener("click", async () => {
  const dataUrl = canvas.toDataURL("image/png");

  predictBtn.disabled = true;
  predictBtn.textContent = "Grading...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: dataUrl, mode: currentMode }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction failed");
    }

    showResult(data);
  } catch (err) {
    resultEmpty.hidden = true;
    resultFilled.hidden = true;
    resultError.hidden = false;
    errorText.textContent = err.message;
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Grade it";
  }
});

function showResult(data) {
  resultEmpty.hidden = true;
  resultError.hidden = true;
  resultFilled.hidden = false;

  predictedDigitEl.textContent = data.prediction;
  confidenceValueEl.textContent = `${(data.confidence * 100).toFixed(1)}%`;

  // Re-trigger the circle-draw animation
  const circle = document.querySelector(".circle-svg circle");
  circle.style.animation = "none";
  void circle.offsetWidth;
  circle.style.animation = null;

  const sorted = Object.entries(data.scores)
    .sort((a, b) => Number(b[1]) - Number(a[1]));

  scoresEl.innerHTML = "";
  sorted.forEach(([digit, score]) => {
    const row = document.createElement("div");
    row.className = "score-row" + (digit === String(data.prediction) ? " top" : "");
    row.innerHTML = `
      <span>${digit}</span>
      <span class="score-bar-track">
        <span class="score-bar-fill" style="width:${(score * 100).toFixed(1)}%"></span>
      </span>
      <span>${(score * 100).toFixed(1)}%</span>
    `;
    scoresEl.appendChild(row);
  });
}
