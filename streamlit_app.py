import streamlit as st
import numpy as np
import librosa
import pickle
import tempfile
from tensorflow.keras.models import model_from_json

st.set_page_config(page_title="Speech Emotion Recognition", page_icon="🎙️")

# ---- Load model architecture + weights (cached so it loads once) ----
@st.cache_resource
def load_model():
    with open("model_json.json", "r") as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights("Emotion_Model.h5")
    return model

@st.cache_resource
def load_labels():
    with open("labels", "rb") as f:
        return pickle.load(f)

model = load_model()
lb = load_labels()

# ---- Feature extraction (matches training pipeline) ----
def extract_features(audio_path):
    X, sample_rate = librosa.load(
        audio_path, duration=2.5, sr=22050, offset=0.5
    )
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    return mfccs

def predict_emotion(audio_path):
    feats = extract_features(audio_path)

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
