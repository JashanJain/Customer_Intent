import streamlit as st
import numpy as np
import pickle
import re
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

# IMPORTANT: Use MAX_LEN = 50 (same as training)
MAX_LEN = 50

def preprocess_text(text):
    """Apply the same preprocessing as training"""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def predict_intent(text):
    # Preprocess
    text = preprocess_text(text)
    
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Predict
    prediction = model.predict(padded, verbose=0)
    
    return prediction, label_encoder.classes_

# UI
st.title("🤖 Intent Prediction Model")
st.write("Enter text to predict the intent using LSTM model")

# Text input
text_input = st.text_area("Enter your text:", height=100, placeholder="Type here...")

if st.button("Predict Intent", type="primary"):
    if text_input.strip():
        # Get prediction
        prediction, classes = predict_intent(text_input)
        
        # Display all probabilities
        st.subheader("📊 All Intent Probabilities:")
        for intent_name, prob in zip(classes, prediction[0]):
            st.write(f"**{intent_name}:** {prob*100:.2f}%")
        
        # Top prediction
        predicted_class = np.argmax(prediction, axis=1)
        top_intent = label_encoder.inverse_transform(predicted_class)[0]
        confidence = float(np.max(prediction))
        
        st.success(f"🎯 **Predicted Intent:** {top_intent}")
        st.info(f"📈 **Confidence:** {confidence:.2%}")
    else:
        st.error("Please enter some text!")

# Add examples
st.divider()
st.subheader("💡 Try these examples:")
col1, col2 = st.columns(2)

with col1:
    if st.button("Example 1: Feature Request"):
        st.session_state.example = "I would really love if you could add dark mode"
    if st.button("Example 2: Bug Report"):
        st.session_state.example = "This update completely broke my app"
    if st.button("Example 3: Praise"):
        st.session_state.example = "Thank you so much for the fast support"

with col2:
    if st.button("Example 4: Question"):
        st.session_state.example = "How can I reset my password?"
    if st.button("Example 5: Complaint"):
        st.session_state.example = "The app keeps crashing when I try to upload photos"

# Show selected example
if 'example' in st.session_state:
    st.info(f"Selected: {st.session_state.example}")
