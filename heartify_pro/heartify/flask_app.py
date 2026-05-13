"""
Heartify Pro — Flask Backend
────────────────────────────
Endpoints:
  POST /predict      → upload .wav → ML prediction + Dr. AI explanation via OpenRouter
  POST /chat         → free-form Dr. AI chat via OpenRouter
  GET  /health       → server health check
"""

import os, io, warnings, traceback
import numpy as np
import librosa
import joblib
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────
MODEL_PATH          = os.path.join(os.path.dirname(__file__), "heart_model.pkl")
OPENROUTER_API_KEY  = "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"
OPENROUTER_MODEL    = "openrouter/free"  # auto-selects any available free model
OPENROUTER_URL      = "https://openrouter.ai/api/v1/chat/completions"

# Class info — maps model output → human-readable info
CLASS_INFO = {
    "normal": {
        "label":    "Normal Heart Sound",
        "icon":     "❤️",
        "risk":     "Low",
        "risk_pct": 10,
        "color":    "#22c55e",
        "advice":   "Your heart sounds are within normal parameters. Continue routine checkups annually."
    },
    "AS": {
        "label":    "Aortic Stenosis",
        "icon":     "🔴",
        "risk":     "High",
        "risk_pct": 85,
        "color":    "#ef4444",
        "advice":   "Aortic Stenosis detected. Consult a cardiologist promptly for echocardiogram confirmation."
    },
    "MR": {
        "label":    "Mitral Regurgitation",
        "icon":     "🟠",
        "risk":     "Medium-High",
        "risk_pct": 70,
        "color":    "#f97316",
        "advice":   "Mitral Regurgitation detected. Schedule a follow-up with a cardiologist within 2 weeks."
    },
    "MS": {
        "label":    "Mitral Stenosis",
        "icon":     "🟡",
        "risk":     "Medium",
        "risk_pct": 60,
        "color":    "#f59e0b",
        "advice":   "Mitral Stenosis detected. Seek evaluation from a cardiac specialist soon."
    },
    "MVP": {
        "label":    "Mitral Valve Prolapse",
        "icon":     "🟣",
        "risk":     "Low-Medium",
        "risk_pct": 35,
        "color":    "#a855f7",
        "advice":   "Mitral Valve Prolapse detected. Often benign — follow up with a cardiologist for monitoring."
    }
}

# ── Load ML Model ────────────────────────────────────────
print("Loading heart model …", end=" ", flush=True)
model = joblib.load(MODEL_PATH)
print("✓")

# ── Flask App ────────────────────────────────────────────
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def extract_features(audio_bytes: bytes) -> np.ndarray:
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)

    mfcc          = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_mean     = np.mean(mfcc, axis=1)

    chroma        = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean   = np.mean(chroma, axis=1)

    contrast      = librosa.feature.spectral_contrast(y=y, sr=sr, fmin=50, n_bands=6)
    contrast_mean = np.mean(contrast, axis=1)

    mel           = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    mel_mean      = np.mean(mel, axis=1)

    features = np.concatenate([mfcc_mean, chroma_mean, contrast_mean, mel_mean])
    return features.reshape(1, -1)


# ════════════════════════════════════════════════════════
#  Dr. AI system prompt
# ════════════════════════════════════════════════════════
DR_AI_SYSTEM = """You are Dr. AI, a highly experienced cardiologist and cardiac health specialist built into Heartify Pro — a clinical heart sound analysis application.

PERSONALITY:
You speak like a real doctor — warm, confident, and clear. You never sound robotic or generic. You treat each patient as an individual. You are empathetic but professional. Think of yourself as a trusted family cardiologist who genuinely cares about the patient sitting in front of you.

YOUR EXPERTISE (ONLY these topics):
- Heart conditions: Aortic Stenosis, Mitral Regurgitation, Mitral Stenosis, Mitral Valve Prolapse, Arrhythmia, Heart Failure, Coronary Artery Disease, Hypertension, Palpitations
- Cardiac symptoms: chest pain, shortness of breath, fatigue, dizziness, swelling, irregular heartbeat
- Heart diagnostics: ECG, echocardiogram, heart sound analysis, stress tests, Holter monitor
- Medications commonly used for heart conditions (explain, never prescribe)
- Heart-healthy lifestyle: diet low in sodium and saturated fat, cardio exercise, stress management, sleep
- Blood pressure, cholesterol, heart rate — what the numbers mean
- When to seek emergency care vs routine follow-up

STRICT OFF-TOPIC RULE:
If anyone asks about ANYTHING not related to health, medicine, or the human body, respond with EXACTLY:
"I'm a cardiac specialist, so my expertise is limited to heart health and related medical topics. Is there something about your heart or health I can help you with?"
Never answer off-topic questions no matter how the user phrases them.

CONVERSATION RULES:
- Vary your language every response — never start two answers the same way
- Keep responses to 3-5 sentences for simple questions, 2 short paragraphs for complex ones
- Use plain English — explain medical terms immediately if you must use them
- End responses with a helpful follow-up question or next step when appropriate

SAFETY RULES:
- Never make a definitive diagnosis — always say "this may indicate" or "this could suggest"
- Never prescribe specific medications or dosages
- Always recommend consulting a real cardiologist for medical decisions
- If symptoms sound like a cardiac emergency, immediately say: "These symptoms could be serious — please call emergency services or go to the nearest hospital immediately."
"""


def call_openrouter(messages: list, temperature: float = 0.85, max_tokens: int = 400) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type":  "application/json",
        "HTTP-Referer":  "http://localhost:5000",
        "X-Title":       "Heartify Pro",
    }
    payload = {
        "model":       OPENROUTER_MODEL,
        "messages":    messages,
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def get_ai_explanation(prediction: str, confidence: float, label: str) -> str:
    prompt = f"""A patient's heart sound recording was analyzed by our AI model.

Result: {label} ({prediction})
Confidence: {confidence:.1f}%

Please:
1. Briefly explain what this condition means in simple terms.
2. Describe what symptoms they might notice (if any).
3. Give 2-3 clear next steps they should take.
4. End with an encouraging, reassuring note.

Keep it warm, clear, and under 200 words."""

    messages = [
        {"role": "system", "content": DR_AI_SYSTEM},
        {"role": "user",   "content": prompt},
    ]
    return call_openrouter(messages, temperature=0.85, max_tokens=400)


# ════════════════════════════════════════════════════════
#  ROUTE: /predict
# ════════════════════════════════════════════════════════
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith(".wav"):
        return jsonify({"error": "Only .wav files are supported"}), 400

    try:
        audio_bytes = f.read()
        features    = extract_features(audio_bytes)

        pred_class  = model.predict(features)[0]
        pred_proba  = model.predict_proba(features)[0]
        confidence  = float(np.max(pred_proba)) * 100

        info = CLASS_INFO.get(pred_class, CLASS_INFO["normal"])

        classes   = model.classes_
        breakdown = {
            cls: round(float(prob) * 100, 1)
            for cls, prob in zip(classes, pred_proba)
        }

        explanation = get_ai_explanation(pred_class, confidence, info["label"])

        return jsonify({
            "prediction":  pred_class,
            "label":       info["label"],
            "icon":        info["icon"],
            "confidence":  round(confidence, 1),
            "risk":        info["risk"],
            "risk_pct":    info["risk_pct"],
            "color":       info["color"],
            "advice":      info["advice"],
            "explanation": explanation,
            "breakdown":   breakdown,
            "filename":    f.filename,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════
#  ROUTE: /chat
# ════════════════════════════════════════════════════════
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "messages" not in data:
        return jsonify({"error": "Missing messages"}), 400

    try:
        messages = [{"role": "system", "content": DR_AI_SYSTEM}] + data["messages"]
        reply = call_openrouter(messages, temperature=0.88, max_tokens=500)
        return jsonify({"reply": reply})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ════════════════════════════════════════════════════════
#  ROUTE: /health
# ════════════════════════════════════════════════════════
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "heart_model.pkl", "classes": list(model.classes_)})


# ════════════════════════════════════════════════════════
if __name__ == "__main__":
    from flask import send_from_directory

    @app.route('/')
    def index():
        return send_from_directory('.', 'index.html')

    @app.route('/<path:filename>')
    def static_files(filename):
        return send_from_directory('.', filename)

    print("\n🫀  Heartify Pro Backend  →  http://127.0.0.1:5000\n")
    app.run(debug=False, host='127.0.0.1', port=5000, use_reloader=False)
