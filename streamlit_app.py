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

# Display model info
st.sidebar.title("Model Info")
st.sidebar.write(f"**Model Output Shape:** {model.output_shape}")
st.sidebar.write(f"**Number of Classes:** {model.output_shape[-1]}")
st.sidebar.write(f"**Label Classes:** {list(label_encoder.classes_)}")
st.sidebar.write(f"**Tokenizer Vocab Size:** {len(tokenizer.word_index)}")

# IMPORTANT: Use MAX_LEN = 50 (same as training)
MAX_LEN = 50

def preprocess_text(text):
    """Apply the same preprocessing as training"""
    original = text
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Also apply stopwords removal and lemmatization like in training
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
    
    try:
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        tokens = text.split()
        tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
        text = " ".join(tokens)
    except:
        # If NLTK data not available, skip this step
        pass
    
    return text, original

def predict_intent(text):
    # Preprocess
    processed_text, original_text = preprocess_text(text)
    
    # Tokenize and pad
    sequence = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(sequence, maxlen=MAX_LEN, padding='post', truncating='post')
    
    # Predict
    prediction = model.predict(padded, verbose=0)
    
    return prediction, label_encoder.classes_, processed_text, original_text, sequence, padded

# UI
st.title("🤖 Intent Prediction Model")
st.write("Enter text to predict the intent using LSTM model")

# Text input
text_input = st.text_area("Enter your text:", height=100, placeholder="Type here...")

# Debug mode
debug_mode = st.checkbox("Show Debug Info")

if st.button("Predict Intent", type="primary"):
    if text_input.strip():
        # Get prediction
        prediction, classes, processed_text, original_text, sequence, padded = predict_intent(text_input)
        
        # Debug information
        if debug_mode:
            st.subheader("🔍 Debug Information:")
            st.write(f"**Original Text:** {original_text}")
            st.write(f"**Processed Text:** {processed_text}")
            st.write(f"**Tokenized Sequence:** {sequence}")
            st.write(f"**Padded Shape:** {padded.shape}")
            st.write(f"**First 10 tokens:** {padded[0][:10]}")
        
        # Display all probabilities
        st.subheader("📊 All Intent Probabilities:")
        
        # Create a dataframe for better visualization
        import pandas as pd
        prob_df = pd.DataFrame({
            'Intent': classes,
            'Probability': prediction[0] * 100
        })
        prob_df = prob_df.sort_values('Probability', ascending=False)
        
        # Display as bars
        st.bar_chart(prob_df.set_index('Intent'))
        
        # Display as table
        for intent, prob in zip(prob_df['Intent'], prob_df['Probability']):
            st.write(f"**{intent}:** {prob:.2f}%")
        
        # Top prediction
        predicted_class = np.argmax(prediction, axis=1)
        top_intent = label_encoder.inverse_transform(predicted_class)[0]
        confidence = float(np.max(prediction))
        
        # Color code based on confidence
        if confidence > 0.7:
            st.success(f"🎯 **Predicted Intent:** {top_intent}")
            st.success(f"📈 **Confidence:** {confidence:.2%}")
        elif confidence > 0.4:
            st.warning(f"🎯 **Predicted Intent:** {top_intent}")
            st.warning(f"📈 **Confidence:** {confidence:.2%} (Low confidence)")
        else:
            st.error(f"🎯 **Predicted Intent:** {top_intent}")
            st.error(f"📈 **Confidence:** {confidence:.2%} (Very low confidence)")
            
        # Show raw predictions
        if debug_mode:
            st.subheader("Raw Prediction Values:")
            st.write(prediction[0])
    else:
        st.error("Please enter some text!")

# Add examples
st.divider()
st.subheader("💡 Try these examples:")

examples = {
    "Feature Request": "I would really love if you could add dark mode",
    "Bug Report": "This update completely broke my app and keeps crashing",
    "Praise": "Thank you so much for the fast support, you guys are amazing",
    "Question": "How can I reset my password? I forgot it",
    "Complaint": "This is terrible service, the app never works properly"
}

cols = st.columns(3)
for idx, (label, text) in enumerate(examples.items()):
    with cols[idx % 3]:
        if st.button(f"📝 {label}"):
            st.session_state.example = text

# Show selected example
if 'example' in st.session_state:
    st.info(f"Selected: {st.session_state.example}")
    
# Add a section to test with multiple inputs at once
with st.expander("🔬 Batch Test Mode"):
    st.write("Test multiple sentences at once:")
    test_sentences = [
        "I would love a dark mode feature",
        "The app crashes all the time",
        "Great work, thanks!",
        "How do I change my settings?",
        "This is not working at all"
    ]
    
    if st.button("Run Batch Test"):
        results = []
        for sentence in test_sentences:
            pred, classes, _, _, _, _ = predict_intent(sentence)
            top_intent = label_encoder.inverse_transform([np.argmax(pred)])[0]
            confidence = float(np.max(pred))
            results.append({
                'Text': sentence,
                'Predicted Intent': top_intent,
                'Confidence': f"{confidence:.2%}"
            })
        
        import pandas as pd
        st.dataframe(pd.DataFrame(results))
