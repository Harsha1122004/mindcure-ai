import os
import re
import uuid
import logging
from datetime import datetime, timezone
from functools import wraps

from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

from transformers import pipeline
from cryptography.fernet import Fernet
import firebase_admin
from firebase_admin import credentials, firestore
from dotenv import load_dotenv

# ── Load Environment Variables
load_dotenv()

# ── ENV
PORT = int(os.getenv("PORT", "5000"))
ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")
SERVICE_ACCOUNT_PATH = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "serviceAccountKey.json")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3:mini")  # ✅ Change to qwen2.5:3b-instruct if you want

# ── Encryption Key Validation
if not ENCRYPTION_KEY:
    print("⚠️ ENCRYPTION_KEY missing — generating temp key for dev.")
    ENCRYPTION_KEY = Fernet.generate_key().decode()

fernet = Fernet(ENCRYPTION_KEY.encode())

# ── Flask / CORS / Logging
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})
logging.getLogger("werkzeug").setLevel(logging.WARNING)
log = app.logger

# ✅ JSON Required Decorator
def require_json(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not request.is_json:
            return jsonify({"error": "Expected application/json"}), 400
        return f(*args, **kwargs)
    return wrapper

# ✅ Role Guard Decorator
def require_role(allowed):
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            role = (request.headers.get("X-User-Role") or "").strip().lower()
            if role not in [r.lower() for r in allowed]:
                return jsonify({"error": "Access denied: insufficient role"}), 403
            return f(*args, **kwargs)
        return wrapper
    return decorator

# ── Firebase Init
if not firebase_admin._apps:
    if not os.path.exists(SERVICE_ACCOUNT_PATH):
        raise RuntimeError(f"Service account JSON not found at '{SERVICE_ACCOUNT_PATH}'.")
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)
db = firestore.client()
COLL_SESSIONS = "sessions"
COLL_ALERTS = "crisis_alerts"

# ── NLP Models
sentiment_pipe = pipeline("sentiment-analysis")
emotion_pipe = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base",
    top_k=None
)

# ── Crisis Patterns
CRISIS_PATTERNS = [
    r"\bi\s*(?:want|wish|plan)\s*to\s*(?:die|kill myself|end it|end my life)\b",
    r"\bi\s*(?:can.?t|can't|cannot)\s*go\s*on\b",
    r"\bi\s*hate\s*my\s*life\b",
    r"\bi\s*am\s*(?:suicidal|worthless|hopeless)\b",
    r"\bno\s*reason\s*to\s*live\b",
    r"\bself[- ]?harm\b",
]

ALLOWED_TOPICS = [
    "stress", "anxiety", "mental", "depression", "exam", "focus", "motivation",
    "confidence", "overthinking", "time", "burnout", "lonely", "relationship",
    "self-esteem", "mindset", "sad", "hopeless", "panic", "emotion", "fear",
    "pressure", "study", "worry", "help", "tired", "well-being", "counseling",
    "sleep", "insomnia", "panic attack", "overwhelmed", "overwhelm", "angry", "anger"
]

# ── Helpers
def encrypt_text(s: str) -> str:
    return fernet.encrypt(s.encode()).decode()

def decrypt_text(token: str) -> str:
    return fernet.decrypt(token.encode()).decode()

def ensure_session(session_id: str | None) -> str:
    if not session_id:
        session_id = str(uuid.uuid4())
    ref = db.collection(COLL_SESSIONS).document(session_id)
    if not ref.get().exists:
        ref.set({
            "session_id": session_id,
            "created_at": firestore.SERVER_TIMESTAMP,
            "messages": [],
            "sentiment_trend": [],
            "emotion_trend": [],
        })
    return session_id

def detect_crisis(text: str) -> dict:
    t = text.lower()
    rule_hit = any(re.search(p, t) for p in CRISIS_PATTERNS)
    severity = "high" if rule_hit else "none"
    return {"is_crisis": rule_hit, "severity": severity}

def top_emotions(emotion_scores, top_k=4):
    sorted_labels = sorted(emotion_scores, key=lambda x: x["score"], reverse=True)
    return {e["label"]: round(float(e["score"]), 4) for e in sorted_labels[:top_k]}

def emotion_bucket(scores: dict) -> str:
    if not scores:
        return "neutral"
    norm = {
        "sadness": "sadness", "grief": "sadness", "disappointment": "sadness",
        "worry": "anxiety", "fear": "anxiety", "nervousness": "anxiety", "anxiety": "anxiety",
        "anger": "anger", "annoyance": "anger", "disgust": "anger",
        "joy": "joy", "contentment": "joy", "love": "joy", "admiration": "joy"
    }
    best, best_score = "neutral", 0.0
    for k, v in scores.items():
        mapped = norm.get(k.lower())
        if mapped and v > best_score:
            best, best_score = mapped, v
    return best if best_score >= 0.35 else "neutral"

def is_mental_health_related(text: str) -> bool:
    return any(k in text.lower() for k in ALLOWED_TOPICS)

def ts_now():
    return datetime.now(timezone.utc).isoformat()

def get_history(session_id: str):
    ref = db.collection(COLL_SESSIONS).document(session_id)
    snap = ref.get()
    if not snap.exists:
        return [], ref
    data = snap.to_dict()
    return data.get("messages", []), ref

def transcript_for_llm(messages, max_turns=12):
    conversation = []
    for m in messages[-max_turns:]:
        try:
            text = decrypt_text(m["content"])
        except Exception:
            text = ""
        if not text:
            continue
        role = "User" if m["role"] == "user" else "Assistant"
        conversation.append(f"{role}: {text}")
    return "\n".join(conversation)

# ✅ Intent Detection
def detect_intent_topic(text: str):
    text = text.lower()
    intent = "share"
    if any(k in text for k in ["solution", "what should i", "help", "how do i"]) or text.endswith("?"):
        intent = "request_solution"
    elif len(text.split()) < 5 and any(w in text for w in ["yes", "ok", "sure", "yeah"]):
        intent = "followup"

    topic = "general"
    score_best = 0
    TOPIC_KEYWORDS = {
        "exam_stress": ["exam", "test", "deadline", "study", "marks"],
        "relationships": ["friend", "family", "breakup", "fight"],
        "sleep": ["sleep", "tired", "insomnia"],
        "anxiety_panic": ["panic", "worry", "nervous"],
        "mood_low": ["depress", "sad", "hopeless"]
    }
    for name, kws in TOPIC_KEYWORDS.items():
        s = sum(1 for k in kws if k in text)
        if s > score_best:
            topic, score_best = name, s
    return intent, topic

# ✅ Local LLM Integration
def local_llm_reply(history_text, intent, topic):
    prompt = f"""
You are MindMate, a supportive and empathetic student counselor chatbot.
Your job is to listen deeply, encourage the user, and give 2-3 practical next steps.
NEVER give medical advice. Ask gentle follow-up questions.

Conversation so far:
{history_text}

Intent: {intent}
Topic: {topic}

Respond now:
"""
    try:
        res = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": "phi3:mini", "prompt": prompt, "stream": False},  # ✅ stream: False
            timeout=60
        )
        res.raise_for_status()
        data = res.json()
        reply = data.get("response") or data.get("output") or ""
        return reply.strip() if reply else "I’m here for you. Can you share more?"
    except Exception as e:
        log.error(f"❌ LLM error: {e}")
        return "I’m here with you. Can you share a bit more about how you're feeling?"

# ── Routes
@app.route("/api/session/start", methods=["POST"])
def start_session():
    session_id = ensure_session(None)
    return jsonify({"sessionId": session_id})

@app.route("/api/chat", methods=["POST"])
@require_json
def chat():
    payload = request.get_json() or {}
    session_id = ensure_session(payload.get("sessionId"))
    message = (payload.get("message") or "").strip()

    if not message:
        return jsonify({"error": "Message is required"}), 400

    if not is_mental_health_related(message):
        return jsonify({
            "reply": "💡 I’m here for your mental well-being. Want to share how you're feeling?",
            "sessionId": session_id,
            "emotion": "neutral",
            "analysis": {}
        })

    # Save user message
    history, ref = get_history(session_id)
    ref.update({
        "messages": firestore.ArrayUnion([{
            "at": ts_now(), "role": "user", "content": encrypt_text(message)
        }])
    })

    # Emotion Analysis
    try:
        sent_res = sentiment_pipe(message)[0]
        emo_all = emotion_pipe(message)[0]
        emo_top = top_emotions(emo_all, top_k=4)
        ui_emote = emotion_bucket(emo_top)
    except Exception as e:
        log.error(f"NLP error: {e}")
        sent_res, emo_top, ui_emote = {"label": "NEUTRAL", "score": 0.0}, {}, "neutral"

    # Crisis Detection
    crisis = detect_crisis(message)
    if crisis["is_crisis"]:
        crisis_reply = (
            "I’m really sorry you’re feeling this way. Your life matters. 💙 "
            "If you’re in immediate danger, call your local emergency number. Would you like crisis support resources now?"
        )
        ref.update({"messages": firestore.ArrayUnion([{
            "at": ts_now(), "role": "assistant", "content": encrypt_text(crisis_reply)
        }] )})
        return jsonify({
            "reply": crisis_reply,
            "emotion": "sadness",
            "sessionId": session_id,
            "analysis": {"sentiment": sent_res, "topEmotions": emo_top, "crisis": crisis, "emotion": "sadness"},
        })

    # Intent & Topic
    intent, topic = detect_intent_topic(message)

    # 🧠 Generate reply using local LLM
    history_text = transcript_for_llm(history + [{"role": "user", "content": encrypt_text(message)}])
    final_reply = local_llm_reply(history_text, intent, topic)

    # Store reply
    ref.update({
        "messages": firestore.ArrayUnion([{
            "at": ts_now(), "role": "assistant", "content": encrypt_text(final_reply)
        }])
    })

    return jsonify({
        "reply": final_reply,
        "emotion": ui_emote,
        "sessionId": session_id,
        "analysis": {
            "sentiment": sent_res,
            "topEmotions": emo_top,
            "crisis": crisis,
            "emotion": ui_emote,
            "intent": intent,
            "topic": topic
        },
    })

# ✅ Health Check Endpoint
@app.route("/api/health", methods=["GET"])
def health_check():
    try:
        test_res = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": "hello", "stream": False},
            timeout=10
        )
        test_res.raise_for_status()
        llm_ok = True
    except Exception:
        llm_ok = False

    return jsonify({
        "status": "ok",
        "llm_connected": llm_ok,
        "firestore_connected": True
    })

# ── Start Server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
