import streamlit as st
import tensorflow as tf
import pickle
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and encoders
@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("intent_lstm_model.h5", compile=False)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

model, tokenizer, label_encoder = load_artifacts()

MAX_LEN = 50

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_intent(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    pred = model.predict(padded)
    return label_encoder.inverse_transform([pred.argmax()])[0]

# UI
st.set_page_config(page_title="IntentFlow", page_icon="💬")
st.title("💬 IntentFlow — Customer Feedback Intent Classifier")
st.write("Enter a customer message below and the model will predict the intent.")

user_input = st.text_area("Customer feedback", placeholder="e.g. Please add dark mode to the app")

if st.button("Predict Intent"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        with st.spinner("Predicting..."):
            intent = predict_intent(user_input)
        st.success(f"**Predicted Intent:** {intent}")
