import streamlit as st
import numpy as np
import librosa
import pickle
import tempfile
from tensorflow.keras.models import load_model as keras_load_model

st.set_page_config(
    page_title="Speech Emotion Recognition",
    page_icon="🎙️"
)

# ----------------------------
# Load Model
# ----------------------------
@st.cache_resource
def load_ser_model():
    return keras_load_model("Emotion_Model.h5")

# ----------------------------
# Load Labels
# ----------------------------
@st.cache_resource
def load_labels():
    with open("labels", "rb") as f:
        return pickle.load(f)

model = load_ser_model()
lb = load_labels()

# ----------------------------
# Feature Extraction
# ----------------------------
def extract_features(audio_path):
    X, sample_rate = librosa.load(
        audio_path,
        duration=2.5,
        sr=22050,
        offset=0.5
    )

    mfccs = np.mean(
        librosa.feature.mfcc(
            y=X,
            sr=sample_rate,
            n_mfcc=13
        ),
        axis=0
    )

    return mfccs

# ----------------------------
# Prediction
# ----------------------------
def predict_emotion(audio_path):
    feats = extract_features(audio_path)

    expected_len = model.input_shape[1]

    padded = np.zeros(expected_len)

    n = min(expected_len, len(feats))
    padded[:n] = feats[:n]

    X = padded.reshape(1, expected_len, 1)

    preds = model.predict(X, verbose=0)

    predicted_index = np.argmax(preds)

    label = lb.inverse_transform([predicted_index])[0]

    probs = {
        cls: float(prob)
        for cls, prob in zip(lb.classes_, preds[0])
    }

    return label, probs

# ----------------------------
# UI
# ----------------------------
st.title("🎙️ Speech Emotion Recognition")

st.write(
    "Upload a short speech clip (2–3 seconds) to predict the speaker's emotion."
)

uploaded_file = st.file_uploader(
    "Upload Audio",
    type=["wav", "mp3"]
)

if uploaded_file is not None:

    suffix = "." + uploaded_file.name.split(".")[-1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.read())
        audio_path = tmp.name

    st.audio(uploaded_file)

    with st.spinner("Predicting Emotion..."):
        emotion, probabilities = predict_emotion(audio_path)

    st.success(f"### Predicted Emotion: {emotion}")

    st.subheader("Prediction Probabilities")

    st.bar_chart(probabilities)
    expected_len = model.input_shape[1]
    padded = np.zeros(expected_len)
    n = min(expected_len, len(feats))
    padded[:n] = feats[:n]

    X = padded.reshape(1, expected_len, 1)
    preds = model.predict(X)
    label = lb.inverse_transform([np.argmax(preds)])[0]
    probs = {cls: float(p) for cls, p in zip(lb.classes_, preds[0])}
    return label, probs

# ---- UI ----
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload a short speech clip (~2-3 sec) to detect the emotion. CNN model trained on RAVDESS + SAVEE.")

uploaded_file = st.file_uploader("Upload audio (wav/mp3)", type=["wav", "mp3"])

if uploaded_file is not None:
    # Save to a temp file so librosa can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    st.audio(uploaded_file)

    with st.spinner("Predicting..."):
        label, probs = predict_emotion(tmp_path)

    st.subheader(f"Predicted Emotion: **{label}**")
    st.bar_chart(probs)
