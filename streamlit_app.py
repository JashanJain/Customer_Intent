import gradio as gr
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

# Load model and preprocessing files
model = load_model('intent_lstm_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# IMPORTANT: Use the same MAX_LEN as training (50, not 20!)
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
    
    # Get all probabilities
    result = "**All Intent Probabilities:**\n\n"
    for intent_name, prob in zip(label_encoder.classes_, prediction[0]):
        result += f"• {intent_name}: {prob*100:.2f}%\n"
    
    # Top prediction
    predicted_class = np.argmax(prediction, axis=1)
    top_intent = label_encoder.inverse_transform(predicted_class)[0]
    top_confidence = float(np.max(prediction))
    
    result += f"\n**🎯 Predicted Intent:** {top_intent}\n"
    result += f"**📊 Confidence:** {top_confidence:.2%}"
    
    return result

# Create Gradio interface
iface = gr.Interface(
    fn=predict_intent,
    inputs=gr.Textbox(lines=3, placeholder="Enter your text here..."),
    outputs=gr.Textbox(label="Prediction Results"),
    title="🤖 Intent Prediction Model",
    description="Enter text to predict the intent using LSTM model. The model can identify: Bug, Complaint, Feature Request, Praise, and Question.",
    examples=[
        ["I would really love if you could add dark mode"],
        ["This update completely broke my app"],
        ["Thank you so much for the fast support"],
        ["How can I reset my password?"],
        ["The app keeps crashing when I try to upload photos"]
    ]
)

iface.launch()
