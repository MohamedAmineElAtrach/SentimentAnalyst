import logging
from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS from flask_cors
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re

# Set up logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes in the app

# Load the trained model and vectorizer
with open('trained_model.sav', 'rb') as file:
    loaded_model, vectorizer = pickle.load(file)

# Define the stemming function
def stemming(content):
    # Initialize the Porter stemmer
    port_stem = PorterStemmer()
    # Perform stemming
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)
    return stemmed_content

@app.route('/classify_sentiment', methods=['POST'])  # Allow POST requests only
def classify_sentiment_api():
    if request.method == 'POST':
        data = request.get_json()
        sentence = data.get('sentence')
        logging.debug('Received sentence: %s', sentence)  # Log the received sentence
        if sentence:
            sentiment = classify_sentiment(sentence)
            logging.debug('Sentiment analysis result: %s', sentiment)  # Log the sentiment analysis result
            return jsonify({'sentiment': sentiment}), 200
        else:
            return jsonify({'error': 'Sentence not provided'}), 400
    else:
        return jsonify({'error': 'Method not allowed'}), 405  # Return error for other methods

def classify_sentiment(sentence):
    # Preprocess the input sentence
    preprocessed_sentence = stemming(sentence)
    # Vectorize the preprocessed sentence
    vectorized_sentence = vectorizer.transform([preprocessed_sentence])
    # Ensure that the vectorizer has the same number of features as the model's input
    vectorized_sentence = vectorized_sentence[:, :loaded_model.coef_.shape[1]]
    # Classify sentiment using the loaded model
    prediction = loaded_model.predict(vectorized_sentence)

    if prediction[0] == 0:
        return 'Negative'
    else:
        return 'Positive'

if __name__ == '__main__':
    app.run(debug=True)
