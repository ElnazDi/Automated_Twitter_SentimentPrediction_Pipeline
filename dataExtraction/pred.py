# prediction
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
import pandas as pd
import pickle
import tensorflow as tf
import os


if __name__ == '__main__': 

    # Read the model under my_model folder
    model = load_model("../model")    
    print("model loaded")

    # Read Tweets from Mongo DB  


    result = []
    result.append(input("enter a text"))
    #text_in = input_t
    tokenizer = Tokenizer(num_words=500, split=' ')
    tokenizer.fit_on_texts(result)
    seq = tokenizer.texts_to_sequences(result)
    X = pad_sequences(seq,23)
    pred = model.predict(X)
    print("pred",pred)
    labels = ['empty', 'sadness', 'enthusiasm' ,'neutral', 'worry' ,'surprise', 'love', 'fun',
                'hate', 'happiness', 'boredom' ,'relief', 'anger']

    sentiment_class=labels[np.argmax(pred)]
    print(sentiment_class)
