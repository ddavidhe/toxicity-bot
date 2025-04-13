import tensorflow as tf
from tensorflow import keras
from keras.layers import TextVectorization
import pandas as pd
import os

# An easy way to interact with the model through the terminal

# Load the model
model = tf.keras.models.load_model('toxic.keras')

# Define TextVectorization layer
max_features = 200000
maxlen = 1800
vectorizer = TextVectorization(max_tokens=max_features, output_sequence_length=maxlen)
df = pd.read_csv(os.path.join('csvFiles', 'train.csv', 'train.csv'))
text_samples = df['comment_text'].values
vectorizer.adapt(text_samples)


# Preprocessing function
def preprocess_text(text, vectorizer):
    return vectorizer([text])

categories = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'racist']

# Input from user
while True:
    try:
        input_text = input("Enter a string: ")

        # Preprocess the input
        processed_input = preprocess_text(input_text, vectorizer)

        # Predict using the model
        prediction = model.predict(processed_input)[0]
        result = [categories[i] for i, score in enumerate(prediction) if score > 0.5]
        result_dict = {category: category in result for category in categories}
        for category, is_present in result_dict.items():
            print(f"'{category}': {is_present}")

        # Output the result
        rounded_prediction = [round(score, 3) for score in prediction]
        print("Toxicity Score:", rounded_prediction)
    except KeyboardInterrupt:
        print("\nExiting...")
        break