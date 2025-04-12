import streamlit as st
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Set Streamlit page config
st.set_page_config(page_title="Fake News Classifier", layout="centered")

# Load the trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

# Load the tokenizer
@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()
tokenizer = load_tokenizer()

# Function to preprocess and predict
def classify_news(text):
    max_len = 300  # Use the same max length used during training
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    prediction = model.predict(padded)[0][0]
    label = "Fake" if prediction > 0.5 else "Real"
    confidence = prediction if prediction > 0.5 else 1 - prediction
    return label, confidence

# Streamlit UI
st.title("📰 Fake News Classifier")
st.write("Enter a news article to classify it as real or fake.")

user_input = st.text_area("News Article", height=200)

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        label, confidence = classify_news(user_input)
        st.success(f"Prediction: **{label}**")
        st.info(f"Confidence: {confidence:.2%}")
