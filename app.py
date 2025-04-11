import streamlit as st
import nltk
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load model and tokenizer
model = tf.keras.models.load_model('model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

lemmatizer = WordNetLemmatizer()

# Streamlit UI
st.title("📧 Phishing Email Classifier")
st.write("Paste an email and click 'Predict' to check if it's a phishing attempt.")

email_input = st.text_area("Enter the email content here:")

if st.button("Predict"):
    if email_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Preprocess input
        tokens = email_input.split()
        tokens = [word.lower() for word in tokens if word.isalpha()]
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        processed_text = ' '.join(tokens)

        sequence = tokenizer.texts_to_sequences([processed_text])
        padded = pad_sequences(sequence, maxlen=100)

        prediction = model.predict(padded)[0][0]
        result = "🚨 Phishing Email" if prediction > 0.5 else "✅ Legitimate Email"
        st.success(f"Prediction: {result} (Confidence: {prediction:.2f})")
