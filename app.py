from flask import Flask, request, jsonify, render_template
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load model and preprocessing files
print("Loading model...")
model = load_model('intent_lstm_model.h5')

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

print("Model loaded successfully!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data['text']
        
        # Preprocess
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=20)
        
        # Predict
        prediction = model.predict(padded, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)
        
        # Decode
        intent = label_encoder.inverse_transform(predicted_class)[0]
        confidence = float(np.max(prediction))
        
        return jsonify({
            'intent': intent,
            'confidence': f'{confidence:.2%}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)