import streamlit as st
import tensorflow as tf
import pickle
import re
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------
# Config
# -------------------------
MAX_LEN = 50
VOCAB_SIZE = 50000

st.set_page_config(page_title="IntentFlow", page_icon="💬", layout="centered")

# -------------------------
# Load artifacts
# -------------------------
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("intent_lstm_model.h5", compile=False)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_artifacts()

# -------------------------
# Preprocessing
# -------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    return padded

def predict(text):
    padded = preprocess(text)
    probs = model.predict(padded)[0]
    idx = np.argmax(probs)
    intent = label_encoder.inverse_transform([idx])[0]
    confidence = probs[idx]
    return intent, confidence, probs

# -------------------------
# UI
# -------------------------
st.title("💬 IntentFlow")
st.subheader("Customer Feedback Intent Classifier")
st.write("Enter a customer message below to predict its intent using an LSTM-based NLP model.")

example = st.selectbox(
    "Try an example:",
    [
        "",
        "The app crashes when I upload a photo",
        "Can you please add dark mode?",
        "Great work on the new update!",
        "How do I reset my password?"
    ]
)

user_input = st.text_area("Customer feedback", value=example, height=120)

if st.button("🔍 Predict Intent"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            intent, confidence, probs = predict(user_input)

        st.success(f"**Predicted Intent:** {intent}")
        st.info(f"**Confidence:** {confidence:.2%}")

        with st.expander("🔎 See all class probabilities"):
            for label, p in zip(label_encoder.classes_, probs):
                st.write(f"{label}: {p:.2%}")

# -------------------------
# Footer
# -------------------------
st.markdown("---")
st.caption("Built with TensorFlow, NLP preprocessing, and deployed on Streamlit Cloud.")
