{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "X5zw-TrN5XDi",
        "outputId": "4343d3ce-9fef-4794-d16a-f803ad82b009"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Requirement already satisfied: pandas in /usr/local/lib/python3.10/dist-packages (2.1.4)\n",
            "Requirement already satisfied: scikit-learn in /usr/local/lib/python3.10/dist-packages (1.3.2)\n",
            "Requirement already satisfied: nltk in /usr/local/lib/python3.10/dist-packages (3.8.1)\n",
            "Requirement already satisfied: numpy<2,>=1.22.4 in /usr/local/lib/python3.10/dist-packages (from pandas) (1.26.4)\n",
            "Requirement already satisfied: python-dateutil>=2.8.2 in /usr/local/lib/python3.10/dist-packages (from pandas) (2.8.2)\n",
            "Requirement already satisfied: pytz>=2020.1 in /usr/local/lib/python3.10/dist-packages (from pandas) (2024.2)\n",
            "Requirement already satisfied: tzdata>=2022.1 in /usr/local/lib/python3.10/dist-packages (from pandas) (2024.1)\n",
            "Requirement already satisfied: scipy>=1.5.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (1.13.1)\n",
            "Requirement already satisfied: joblib>=1.1.1 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (1.4.2)\n",
            "Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.10/dist-packages (from scikit-learn) (3.5.0)\n",
            "Requirement already satisfied: click in /usr/local/lib/python3.10/dist-packages (from nltk) (8.1.7)\n",
            "Requirement already satisfied: regex>=2021.8.3 in /usr/local/lib/python3.10/dist-packages (from nltk) (2024.5.15)\n",
            "Requirement already satisfied: tqdm in /usr/local/lib/python3.10/dist-packages (from nltk) (4.66.5)\n",
            "Requirement already satisfied: six>=1.5 in /usr/local/lib/python3.10/dist-packages (from python-dateutil>=2.8.2->pandas) (1.16.0)\n"
          ]
        }
      ],
      "source": [
        "!pip install pandas scikit-learn nltk"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.feature_extraction.text import TfidfVectorizer\n",
        "from sklearn.linear_model import LogisticRegression\n",
        "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n",
        "import nltk\n",
        "from nltk.corpus import stopwords\n",
        "from nltk.stem import WordNetLemmatizer\n",
        "import os\n",
        "import tarfile\n",
        "import urllib.request\n",
        "\n",
        "# Download necessary NLTK data\n",
        "nltk.download('stopwords')\n",
        "nltk.download('wordnet')\n",
        "\n",
        "# Download the dataset\n",
        "url = 'https://spamassassin.apache.org/old/publiccorpus/20030228_spam.tar.bz2'\n",
        "urllib.request.urlretrieve(url, 'spam.tar.bz2')\n",
        "url = 'https://spamassassin.apache.org/old/publiccorpus/20030228_easy_ham.tar.bz2'\n",
        "urllib.request.urlretrieve(url, 'ham.tar.bz2')\n",
        "\n",
        "# Extract the dataset\n",
        "tar = tarfile.open('spam.tar.bz2', 'r:bz2')\n",
        "tar.extractall()\n",
        "tar.close()\n",
        "\n",
        "tar = tarfile.open('ham.tar.bz2', 'r:bz2')\n",
        "tar.extractall()\n",
        "tar.close()\n",
        "\n",
        "def load_emails_from_directory(directory, label):\n",
        "    emails = []\n",
        "    for filename in os.listdir(directory):\n",
        "        with open(os.path.join(directory, filename), 'r', encoding='latin-1') as file:\n",
        "            emails.append({'text': file.read(), 'label': label})\n",
        "    return emails\n",
        "\n",
        "# Loading spam and ham emails\n",
        "spam_emails = load_emails_from_directory('spam', 'phishing')\n",
        "ham_emails = load_emails_from_directory('easy_ham', 'legitimate')\n",
        "\n",
        "# Combining them into a single DataFrame\n",
        "emails = pd.DataFrame(spam_emails + ham_emails)\n",
        "\n",
        "# Applying preprocessing\n",
        "lemmatizer = WordNetLemmatizer()\n",
        "def preprocess_text(text):\n",
        "    tokens = text.split()\n",
        "    tokens = [word for word in tokens if word.isalpha()]\n",
        "    tokens = [word.lower() for word in tokens]\n",
        "    tokens = [word for word in tokens if word not in stopwords.words('english')]\n",
        "    tokens = [lemmatizer.lemmatize(word) for word in tokens]\n",
        "    return ' '.join(tokens)\n",
        "\n",
        "emails['text'] = emails['text'].apply(preprocess_text)\n",
        "\n",
        "# Feature extraction\n",
        "vectorizer = TfidfVectorizer()\n",
        "X = vectorizer.fit_transform(emails['text'])\n",
        "y = emails['label']\n",
        "\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "# Train model\n",
        "model = LogisticRegression()\n",
        "model.fit(X_train, y_train)\n",
        "\n",
        "# Predict\n",
        "y_pred = model.predict(X_test)\n",
        "\n",
        "# Evaluate the model\n",
        "print(f'Accuracy: {accuracy_score(y_test, y_pred)}')\n",
        "print(f'Precision: {precision_score(y_test, y_pred, pos_label=\"phishing\")}')\n",
        "print(f'Recall: {recall_score(y_test, y_pred, pos_label=\"phishing\")}')\n",
        "print(f'F1 Score: {f1_score(y_test, y_pred, pos_label=\"phishing\")}')\n",
        "\n",
        "# Predict new emails\n",
        "def predict_email(email_text):\n",
        "    email_text = preprocess_text(email_text)\n",
        "    email_vector = vectorizer.transform([email_text])\n",
        "    prediction = model.predict(email_vector)\n",
        "    return prediction[0]\n",
        "\n",
        "# Example usage\n",
        "new_email = \"Congratulations! You've won a free prize. Click here to claim.\"\n",
        "print(f'This email is: {predict_email(new_email)}')"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "IeGviO1hDtmh",
        "outputId": "c60e6449-25bc-41d6-8635-96b69a18df82"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Unzipping corpora/stopwords.zip.\n",
            "[nltk_data] Downloading package wordnet to /root/nltk_data...\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: 0.9351081530782029\n",
            "Precision: 1.0\n",
            "Recall: 0.6722689075630253\n",
            "F1 Score: 0.8040201005025126\n",
            "This email is: phishing\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# from here tf dnn starts\n",
        "\n",
        "import pandas as pd\n",
        "import tensorflow as tf\n",
        "from tensorflow.keras.layers import Dense, SimpleRNN, Embedding\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import LabelEncoder\n",
        "from tensorflow.keras.preprocessing.text import Tokenizer\n",
        "from tensorflow.keras.preprocessing.sequence import pad_sequences\n",
        "\n",
        "max_words = 10000  # Maximum number of words to keep, based on word frequency\n",
        "max_len = 100  # Maximum length of all sequences\n",
        "\n",
        "tokenizer = Tokenizer(num_words=max_words)\n",
        "tokenizer.fit_on_texts(emails['text'])\n",
        "sequences = tokenizer.texts_to_sequences(emails['text'])\n",
        "padded_sequences = pad_sequences(sequences, maxlen=max_len)\n",
        "\n",
        "# Step 2: Convert labels to numerical format\n",
        "label_encoder = LabelEncoder()\n",
        "labels = label_encoder.fit_transform(emails['label'])\n",
        "print(label_encoder.inverse_transform([0, 1]))\n",
        "\n",
        "# Step 3: Create TensorFlow datasets\n",
        "X_train, X_test, y_train, y_test = train_test_split(padded_sequences, labels, test_size=0.2, random_state=42)\n",
        "\n",
        "train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))\n",
        "test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))\n",
        "\n",
        "model = tf.keras.models.Sequential()\n",
        "model.add(Embedding(max_words, 128, input_length=max_len))\n",
        "model.add(SimpleRNN(10, activation='relu', input_shape=(max_len,), return_sequences=True)) # Set return_sequences=True for all SimpleRNN layers except the last one\n",
        "model.add(SimpleRNN(20, activation='relu', input_shape=(max_len,), return_sequences=True))\n",
        "model.add(SimpleRNN(30, activation='relu', input_shape=(max_len,), return_sequences=True))\n",
        "model.add(SimpleRNN(15, activation='relu', input_shape=(max_len,)))\n",
        "model.add(Dense(1, activation='sigmoid'))\n",
        "model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])\n",
        "\n",
        "model.fit(train_dataset.batch(32), epochs=5, validation_data=test_dataset.batch(32))\n",
        "\n",
        "y_pred = model.predict(X_test)\n",
        "from sklearn.metrics import accuracy_score, precision_score, f1_score\n",
        "y_pred = (y_pred > 0.5).astype(int)\n",
        "accuracy = accuracy_score(y_test, y_pred)\n",
        "precision = precision_score(y_test, y_pred)\n",
        "f1 = f1_score(y_test, y_pred)\n",
        "\n",
        "print(f'Accuracy: {accuracy}')\n",
        "print(f'Precision: {precision}')\n",
        "print(f'F1 Score: {f1}')\n",
        "\n",
        "def predictor(mail):\n",
        "  sequences = tokenizer.texts_to_sequences(emails['text'])\n",
        "  padded_sequences = pad_sequences(sequences, maxlen=max_len)\n",
        "  prediction = model.predict(padded_sequences)\n",
        "  return prediction[0]\n",
        "\n",
        "new_email = \"Congratulations! You've won a free prize. Click here to claim.\"\n",
        "if predict_email(new_email) > .4:\n",
        "  print(\"Spam\")\n",
        "else:\n",
        "  print(\"Legitimate\")"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "scPwluIrGAwb",
        "outputId": "316d4649-5aaa-4057-f204-652a2f67ccda"
      },
      "execution_count": null,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "['legitimate' 'phishing']\n",
            "Epoch 1/5\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "/usr/local/lib/python3.10/dist-packages/keras/src/layers/core/embedding.py:90: UserWarning: Argument `input_length` is deprecated. Just remove it.\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.10/dist-packages/keras/src/layers/rnn/rnn.py:204: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.\n",
            "  super().__init__(**kwargs)\n"
          ]
        },
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\u001b[1m76/76\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m15s\u001b[0m 107ms/step - accuracy: 0.6816 - loss: 0.6236 - val_accuracy: 0.8020 - val_loss: 0.3350\n",
            "Epoch 2/5\n",
            "\u001b[1m76/76\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m10s\u001b[0m 111ms/step - accuracy: 0.9271 - loss: 0.1658 - val_accuracy: 0.9667 - val_loss: 0.1018\n",
            "Epoch 3/5\n",
            "\u001b[1m76/76\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m7s\u001b[0m 87ms/step - accuracy: 0.9918 - loss: 0.0351 - val_accuracy: 0.9884 - val_loss: 0.0572\n",
            "Epoch 4/5\n",
            "\u001b[1m76/76\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m13s\u001b[0m 119ms/step - accuracy: 0.9983 - loss: 0.0059 - val_accuracy: 0.9817 - val_loss: 0.0992\n",
            "Epoch 5/5\n",
            "\u001b[1m76/76\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m8s\u001b[0m 111ms/step - accuracy: 0.9972 - loss: 0.0111 - val_accuracy: 0.9850 - val_loss: 0.0594\n",
            "\u001b[1m19/19\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m1s\u001b[0m 46ms/step\n",
            "Accuracy: 0.9850249584026622\n",
            "Precision: 0.9824561403508771\n",
            "F1 Score: 0.9613733905579399\n",
            "\u001b[1m1/1\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m2s\u001b[0m 2s/step\n",
            "Spam\n"
          ]
        }
      ]
    }
  ]
}
