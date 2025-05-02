# app.py  —  Heart‑Disease Predictor + CardioConsult‑Bot (Gemini 1.5 Flash)
# ----------------------------------------------------------------------------------
import os, streamlit as st, pandas as pd, joblib
from sklearn.impute import SimpleImputer
from google import genai
from google.genai import types
from google.api_core.exceptions import GoogleAPIError

# ----------------------------------------------------------------------------------
# 1. Gemini client (google‑genai)
# ----------------------------------------------------------------------------------
API_KEY = "AIzaSyAhiiRyjaIuEaPtpofgLRhca10p2BwxFug"
if not API_KEY:
    st.stop("❌ GEMINI_API_KEY not found — set env var or .streamlit/secrets.toml")

client = genai.Client(api_key=API_KEY)
MODEL  = "gemini-1.5-flash-8b"   # change to gemini-pro if you prefer

# ----------------------------------------------------------------------------------
# 2. Load ML model & scaler
# ----------------------------------------------------------------------------------
best_model = joblib.load("best_model.pkl")
scaler     = joblib.load("scaler.pkl")

# ----------------------------------------------------------------------------------
# 3. Lookup tables
# ----------------------------------------------------------------------------------
sex_map   = {"Male": 1, "Female": 0}
cp_map    = {"atypical angina": 0, "asymptomatic": 1, "non-anginal": 2, "typical angina": 3}
fbs_map   = {"TRUE": 1, "FALSE": 0}
rest_map  = {"left ventricular hypertrophy": 0, "normal": 1, "ST-T wave abnormality": 2}
exang_map = {"TRUE": 1, "FALSE": 0}
slope_map = {"downsloping": 0, "flat": 1, "upsloping": 2}
thal_map  = {"fixed defect": 0, "normal": 1, "reversable defect": 2}

pred_map  = {0:"No Heart Disease",1:"Mild Heart Disease",
             2:"Moderate Heart Disease",3:"Severe Heart Disease",4:"Critical Heart Disease"}

# ----------------------------------------------------------------------------------
# 4. Streamlit UI
# ----------------------------------------------------------------------------------
st.set_page_config(page_title="Heart‑Disease Predictor + CardioConsult‑Bot", page_icon="🫀")
st.title("🫀 Heart‑Disease Risk Predictor  +  CardioConsult‑Bot")
st.caption("Machine‑learning screening + live cardiovascular chat • *Educational only — not medical advice*")

# ---------------------- Prediction form ------------------------------------------
with st.form("prediction"):
    age      = st.number_input("Age", 1, 120, 50)
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", 50, 250, 130)
    chol     = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 200)
    thalch   = st.number_input("Max Heart Rate Achieved (bpm)", 50, 250, 150)
    oldpeak  = st.number_input("ST Depression", 0.0, 10.0, 1.0, format="%.1f")
    ca       = st.number_input("Number of Major Vessels (0–4)", 0, 4, 0)

    sex      = st.selectbox("Sex", list(sex_map))
    cp       = st.selectbox("Chest‑Pain Type", list(cp_map))
    fbs      = st.selectbox("Fasting Blood Sugar >120 mg/dL", list(fbs_map))
    restecg  = st.selectbox("Resting ECG", list(rest_map))
    exang    = st.selectbox("Exercise‑Induced Angina", list(exang_map))
    slope    = st.selectbox("Slope of Peak Exercise ST", list(slope_map))
    thal     = st.selectbox("Thalassemia", list(thal_map))

    submitted = st.form_submit_button("Predict")

# ---------------------- Make prediction ------------------------------------------
if submitted:
    X = pd.DataFrame({
        "age":[age],"sex":[sex_map[sex]],"cp":[cp_map[cp]],"trestbps":[trestbps],
        "chol":[chol],"fbs":[fbs_map[fbs]],"restecg":[rest_map[restecg]],
        "thalch":[thalch],"exang":[exang_map[exang]],"oldpeak":[oldpeak],
        "slope":[slope_map[slope]],"ca":[ca],"thal":[thal_map[thal]]
    })
    if X.shape[1] != scaler.mean_.shape[0]:
        X = X.drop(columns=["thal"])        # scaler trained on 13 features

    X_scaled = scaler.transform(SimpleImputer(strategy="median").fit_transform(X))
    y_pred   = int(best_model.predict(X_scaled)[0])
    pred_txt = pred_map.get(y_pred, "Unknown")

    st.subheader("🔎 Prediction Result")
    st.success(f"**Predicted Heart‑Disease Level:** {pred_txt}")

    st.session_state.prediction_ctx = dict(X.iloc[0], model_prediction=pred_txt)

# ----------------------------------------------------------------------------------
# 5. Chat memory initialisation
# ----------------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        ("assistant",
         "👋 Hi — I'm **CardioConsult‑Bot**. Ask me about your heart‑disease screening result or cardiovascular health. "
         "_I am not a substitute for a doctor; for urgent or non‑cardiac issues, seek professional care._")
    ]

# Display last messages
for role, msg in st.session_state.messages[-8:]:
    st.chat_message(role).markdown(msg)

# ----------------------------------------------------------------------------------
# 6. Chat input -> Gemini stream
# ----------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------
# 6. Chat input -> Gemini (show full response once)
# ----------------------------------------------------------------------------------
if prompt := st.chat_input("Ask about your cardiovascular health…"):
    # 1️⃣ Save user message and show chat history
    st.session_state.messages.append(("user", prompt))

    # Show previous messages (chat history)
    if "messages" in st.session_state:
        for message in st.session_state.messages:
            if message[0] == "user":
                st.chat_message("user").markdown(message[1])
            else:
                st.chat_message("assistant").markdown(message[1])

    # 2️⃣ Build single user prompt (instructions + context + question)
    instructions = (
        "You are CardioConsult‑Bot, an educational assistant who discusses cardiovascular "
        "screening results, risk factors, and heart‑healthy lifestyle. "
        "Do **not** prescribe medications or diagnose conditions outside cardiology. "
        "Urge professional medical advice for emergencies.\n\n"
    )
    context_txt = ""
    if "prediction_ctx" in st.session_state:
        ctx = st.session_state.prediction_ctx
        context_txt = "\n".join(f"{k}: {v}" for k, v in ctx.items()) + "\n\n"

    req_content = [
        types.Content(
            role="user",
            parts=[types.Part(text=instructions + context_txt + prompt)]
        )
    ]

    # Debugging output for `req_content`
    print("Request content:", req_content)

    gen_cfg = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=0),
        response_mime_type="text/plain",
    )

    # 3️⃣ Call Gemini and buffer chunks
    full_reply = ""
    try:
        with st.spinner("Consulting Gemini…"):
            for chunk in client.models.generate_content_stream(
                model=MODEL, contents=req_content, config=gen_cfg
            ):
                if chunk.text:
                    full_reply += chunk.text
    except (GoogleAPIError, Exception) as e:
        full_reply = f"⚠️ Service error: {getattr(e,'message',e)}"

    # 4️⃣ Display the complete answer exactly once
    st.chat_message("assistant").markdown(full_reply)
    st.session_state.messages.append(("assistant", full_reply))
