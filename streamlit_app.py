import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Cache the model loading
@st.cache_resource
def load_models():
    model = load_model('intent_lstm_model.h5')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, tokenizer, label_encoder

# Load models
model, tokenizer, label_encoder = load_models()

# UI
st.title("🤖 Intent Prediction Model")
st.write("Enter text to predict the intent using LSTM model")

text_input = st.text_area("Enter your text:", height=100, placeholder="Type here...")

if st.button("Predict Intent", type="primary"):
    if text_input.strip():
        # Predict
        sequence = tokenizer.texts_to_sequences([text_input])
        padded = pad_sequences(sequence, maxlen=20)
        prediction = model.predict(padded, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)
        intent = label_encoder.inverse_transform(predicted_class)[0]
        confidence = float(np.max(prediction))
        
        # Display results
        st.success(f"**Intent:** {intent}")
        st.info(f"**Confidence:** {confidence:.2%}")
    else:
        st.error("Please enter some text!")
