from flask import Flask, request, render_template
from tensorflow import keras
from keras.layers import TextVectorization
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import os
import numpy as np
import pandas as pd

app = Flask(__name__)
model = load_model('toxic.keras')

categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'racist']
df = pd.read_csv(os.path.join('csvFiles', 'train.csv', 'train.csv'))
X = df['comment_text'].values

max_features = 200000  # number of words in the vacab,  
max_len = 1800  # Sequence length to pad the outputs to.

vectorizer = TextVectorization(max_tokens = max_features, output_sequence_length = max_len, output_mode = 'int')
vectorizer.adapt(X)

# vectorize text first for the model
def preprocess_input(text):
    vectorized_text = vectorizer(text)

    padded = pad_sequences([vectorized_text], maxlen=max_len)
    return padded.astype("int32")


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        user_input = request.form['user_input']
        # Preprocess the input text
        preprocessed_input = preprocess_input(user_input)
        result = model.predict(preprocessed_input)
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
    