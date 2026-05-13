# 🫀 Heartify Pro — Clinical Heart Sound Analysis

![Python](https://img.shields.io/badge/Python-3.x-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-REST%20API-lightgrey?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-KNN-orange?style=flat-square&logo=scikit-learn)
![Gemini](https://img.shields.io/badge/Gemini-2.0%20Flash-green?style=flat-square&logo=google)
![Accuracy](https://img.shields.io/badge/Accuracy-~97%25-brightgreen?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

> AI-powered cardiac heart sound analysis web app. Upload a `.wav` recording, get an instant ML prediction across 5 cardiac conditions, and receive a natural language explanation from a Gemini-powered cardiac AI assistant.

---

## 📁 What You Have

```
📁 heartify/
├── index.html          ← Frontend (7-page single-file app)
├── flask_app.py        ← Backend REST API
├── heart_model.pkl     ← Trained KNN model (Yaseen Khan 2018)
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## ⚙️ How It All Works

```
You (Browser)
    │
    │  1. Open http://127.0.0.1:5000 in Chrome / Edge
    │  2. Upload a .wav heart sound file
    │  3. Click "Analyze Now"
    │
    ▼
index.html  ──POST /predict──▶  flask_app.py  (localhost:5000)
                                    │
                                    ├── librosa extracts 99 audio features
                                    ├── heart_model.pkl predicts condition
                                    └── Gemini API generates Dr. AI explanation
                                    │
                                    └──JSON response──▶  index.html shows results
```

---

## 🚀 Setup & Run

### Step 1 — Install Python Dependencies

```bash
cd heartify
pip install flask flask-cors librosa joblib scikit-learn numpy requests soundfile
```

Or using requirements.txt:
```bash
pip install -r requirements.txt
```

---

### Step 2 — Confirm Model File Is in Place

> ✅ `heart_model.pkl` must be in the **same folder** as `flask_app.py`

```
heartify/
├── flask_app.py
├── heart_model.pkl   ← must be here!
├── index.html
└── requirements.txt
```

---

### Step 3 — Start the Flask Backend

```bash
python flask_app.py
```

Expected output:
```
Loading heart model … ✓
🫀  Heartify Pro Backend  →  http://127.0.0.1:5000
 * Running on http://127.0.0.1:5000
```

> ⚠️ **Keep this terminal open** the entire time you use the app. Closing it shuts down the backend.

---

### Step 4 — Open the App

Flask now serves both the API **and** the frontend. Just open:

```
http://127.0.0.1:5000
```

No second terminal needed.

---

### Step 5 — Use the App

1. Click the **Analysis** page (waveform icon in sidebar)
2. Drop your `.wav` file onto the upload area, or click to browse
3. Click **🫀 Analyze Now**
4. Wait ~3–5 seconds for results
5. View detected condition, confidence score, and Dr. AI explanation
6. Check the **Report** page for the full clinical summary
7. Chat with **Dr. AI** for follow-up questions

---

## 📄 App Pages

| Icon | Page | Description |
|------|------|-------------|
| 🏠 | Overview | Live vitals dashboard, ECG animation, organ health cards, scan history |
| 📡 | Analysis | Upload `.wav` → ML prediction + confidence score + Gemini AI explanation |
| 📈 | Trends | Heart rate, blood pressure, and glucose trend charts |
| 📄 | Report | Full clinical report of latest analysis with printable layout |
| 💬 | Dr. AI Chat | Gemini-powered cardiac specialist chatbot with voice input/output |
| 🔔 | Alerts | Notification log for scan results and follow-up reminders |
| ⚙️ | Settings | Backend URL config, voice toggle, Flask connection status |

---

## ❤️ Heart Conditions Detected

The KNN model classifies recordings into 5 categories:

| Code | Condition | Risk | Description |
|------|-----------|------|-------------|
| `normal` | Normal Heart Sound | 🟢 Low | Healthy S1/S2 — no abnormal murmurs |
| `AS` | Aortic Stenosis | 🔴 High | Systolic ejection murmur — narrowed aortic valve |
| `MR` | Mitral Regurgitation | 🟠 Medium-High | Holosystolic murmur — backflow through mitral valve |
| `MS` | Mitral Stenosis | 🟡 Medium | Diastolic rumble — narrowed mitral valve opening |
| `MVP` | Mitral Valve Prolapse | 🟣 Low-Medium | Mid-systolic click — billowing mitral leaflets |

---

## 🛠️ Tech Stack

| Layer | Technology | Details |
|-------|-----------|---------|
| Frontend | HTML / CSS / JS | Single-file SPA, no framework, 7 pages |
| Backend | Python + Flask | REST API on port 5000, flask-cors |
| ML Model | scikit-learn KNN | Pipeline + StandardScaler, ~97% CV accuracy |
| Audio Features | librosa | 40 MFCC + 12 Chroma + 7 Spectral Contrast + 40 Mel = **99 features** |
| AI Chat | Google Gemini 2.0 Flash | Dr. AI cardiac specialist persona |
| Voice I/O | Web Speech API | SpeechRecognition + SpeechSynthesis (Chrome/Edge only) |
| Android App | Capacitor | Wraps index.html into a native Android APK |

---

## 🌐 API Endpoints

```
GET  /health
     → { status: 'ok', model: 'heart_model.pkl', ai: 'Gemini 2.0 Flash' }

POST /predict
     Body: multipart/form-data { file: <recording.wav> }
     → { prediction, label, confidence, risk, explanation, breakdown }

POST /chat
     Body: { messages: [ { role: 'user', content: '...' } ] }
     → { reply: '<Dr. AI response>' }
```

---

## 🔧 Troubleshooting

**❌ "Failed to fetch" or "Connection Reset"**
→ Flask is not running. Run `python flask_app.py` first.
→ Make sure you open `http://127.0.0.1:5000` (not port 8080).

**❌ "Only .wav files are supported"**
→ Convert your audio to WAV format using Audacity or any online converter.

**⚠️ InconsistentVersionWarning in terminal**
→ Normal warning — the model still works. To silence it:
```bash
pip install scikit-learn==1.6.1
```

**❌ Settings page shows "❌ Offline"**
→ Flask server is not running. Start it and refresh the browser.

**❌ Voice input not working**
→ Voice requires **Chrome or Edge**. Firefox does not support the Web Speech API.
→ Allow microphone access when the browser prompts you.

---

## 📱 Building the Android APK

```bash
# 1. Install Node.js from https://nodejs.org first

# 2. Init and install Capacitor
npm init -y
npm install @capacitor/core @capacitor/cli @capacitor/android

# 3. Init Capacitor
npx cap init "Heartify Pro" "com.heartify.pro" --web-dir .

# 4. Add Android platform
npx cap add android

# 5. Update BACKEND url in index.html to your PC's local IP
#    const BACKEND = 'http://192.168.x.x:5000';

# 6. Update flask_app.py last line to:
#    app.run(debug=False, host='0.0.0.0', port=5000, use_reloader=False)

# 7. Sync and open Android Studio
npx cap sync
npx cap open android

# 8. In Android Studio: Build → Build APK(s)
# APK output: android/app/build/outputs/apk/debug/app-debug.apk
```

---

## 📦 Requirements

```
flask
flask-cors
librosa
joblib
scikit-learn==1.6.1
numpy
requests
soundfile
```

---

## ⚠️ Medical Disclaimer

> Heartify Pro is a **research and educational project only**. It is **NOT** a certified medical device and must **NOT** replace professional medical diagnosis or treatment. Always consult a qualified cardiologist for any cardiac concerns.

---

## 👨‍💻 Author

Built with ❤️ by **Sayan** — Heartify Pro v1.0 — 2026
