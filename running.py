import os
import pandas as pd
import gradio as gr
import tensorflow as tf

from tensorflow import keras
from keras.layers import TextVectorization

df = pd.read_csv(os.path.join('csvFiles','train.csv','train.csv'))


model = tf.keras.models.load_model('toxic.h5')

max_features = 200000
max_len = 1800

vectorizer = TextVectorization(max_tokens = max_features, output_sequence_length = max_len,output_mode = 'int')

input_str = vectorizer('sample text') # input box


def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)

    text = ''
    for idx, col in enumerate(df.columns[[2,5,7]]): # I only really want toxic, insult, and racist
        text += '{}: {}\n'.format(col, results[0][idx] > 0.5)
    
    return text


interface = gr.Interface(fn=score_comment, 
                         inputs=gr.Textbox(lines=2, placeholder='Comment to score'),
                        outputs='text')

interface.launch(share=True)