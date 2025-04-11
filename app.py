import streamlit as st
import pandas as pd
import numpy as np
import joblib
import nltk
import tensorflow as tf

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Ensure necessary NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Preprocessing functions
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    tokens = text.split()
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word.lower() for word in tokens]
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Load the dataset and train models only once
@st.cache_resource
def load_models():
    # Load and preprocess dataset
    import os, tarfile, urllib.request
    def download_and_extract(url, filename):
        urllib.request.urlretrieve(url, filename)
        with tarfile.open(filename, 'r:bz2') as tar:
            tar.extractall()

    if not os.path.exists('spam'):
        download_and_extract("https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2", "spam.tar.bz2")
    if not os.path.exists('easy_ham'):
        download_and_extract("https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2", "ham.tar.bz2")

    def load_emails(directory, label):
        emails = []
        for file in os.listdir(directory):
            with open(os.path.join(directory, file), 'r', encoding='latin-1') as f:
                emails.append({'text': f.read(), 'label': label})
        return emails

    spam = load_emails('spam', 'phishing')
    ham = load_emails('easy_ham', 'legitimate')
    emails = pd.DataFrame(spam + ham)
    emails['text'] = emails['text'].apply(preprocess_text)

    # Logistic Regression model
    vectorizer = TfidfVectorizer()
    X_tfidf = vectorizer.fit_transform(emails['text'])
    y = emails['label']
    log_model = LogisticRegression()
    log_model.fit(X_tfidf, y)

    # RNN Model
    max_words = 10000
    max_len = 100
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(emails['text'])
    sequences = tokenizer.texts_to_sequences(emails['text'])
    padded = pad_sequences(sequences, maxlen=max_len)

    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(max_words, 128),
        tf.keras.layers.SimpleRNN(10, activation='relu', return_sequences=True),
        tf.keras.layers.SimpleRNN(20, activation='relu', return_sequences=True),
        tf.keras.layers.SimpleRNN(30, activation='relu', return_sequences=True),
        tf.keras.layers.SimpleRNN(15, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(padded, y_encoded, epochs=3, batch_size=32, verbose=0)

    return log_model, vectorizer, model, tokenizer, encoder

log_model, vectorizer, rnn_model, tokenizer, label_encoder = load_models()

# Streamlit UI
st.title("📧 Email Spam/Phishing Detection")
st.write("Enter an email message below and let the models determine if it's **phishing/spam** or **legitimate**.")

email_input = st.text_area("📨 Email Content", height=200, value="Congratulations! You've won a prize!")

if st.button("🔍 Classify"):
    cleaned = preprocess_text(email_input)

    # Logistic Regression Prediction
    tfidf_input = vectorizer.transform([cleaned])
    log_pred = log_model.predict(tfidf_input)[0]

    # RNN Prediction
    seq = tokenizer.texts_to_sequences([cleaned])
    pad_seq = pad_sequences(seq, maxlen=100)
    rnn_pred = rnn_model.predict(pad_seq)[0][0]
    rnn_label = label_encoder.inverse_transform([int(rnn_pred > 0.5)])[0]

    st.markdown("### 🔎 Results")
    st.write(f"**Logistic Regression:** `{log_pred}`")
    st.write(f"**RNN Model:** `{rnn_label}` (Confidence: {rnn_pred:.2f})")

    if log_pred == 'phishing' or rnn_label == 'phishing':
        st.error("⚠️ This email seems to be **PHISHING/SPAM**.")
    else:
        st.success("✅ This email seems to be **LEGITIMATE**.")
