# Importing required libraries
import tensorflow as tf
import pandas as pd
import numpy as np

import data_cleaning
from urllib.parse import urlparse
import nltk 

nltk.download('stopwords')
from nltk.corpus import stopwords

from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from keras.optimizers import adam_experimental
from sklearn.model_selection import train_test_split 

import pickle
# import processing



class Emotions():
    '''Predictor Class for ingesting emotions of the tweets'''

    def __init__(self):
        '''Class Attributes: train.txt includes 13 different lables'''
        data = pd.read_csv('train.txt',sep=',',header=0)
        self.df = pd.DataFrame(data)


    def modelling(self):
        '''Class Method: 

        Cleaning: 
        (1) using our own datacleaning method, 
        (2) removing stopwords,

        Lexicanl level: 
        Tokenizing sentences

        Hyperparameters:
        (1) Num-words:  number of words to keep based on the frequency of words.
        (2) Split: separator used for splitting the word

        '''    

        cleaningClass = data_cleaning.DataCleaning()     
        self.df['cleaned_text'] = self.df[['content']].applymap(
            lambda x: cleaningClass.data_preprocessing(x))

        stop_words = stopwords.words('english')
        self.df['cleaned_text'] = self.df['cleaned_text'].apply(lambda x: 
            ' '.join(x for x in x.split() if x not in stop_words))
  
         
        tokenizer = Tokenizer(num_words=500, split=' ') 
        tokenizer.fit_on_texts(self.df['cleaned_text'].values)
        X = tokenizer.texts_to_sequences(self.df['cleaned_text'].values)
        X = pad_sequences(X) 

        dropout = 0.2
        rec_dropout = 0.2
        #customAdam = keras.optimizers.Adam(lr=0.0001)

        # Running the MLflow -> LSTM Model
        model = Sequential()
        model.add(Embedding(500, 120, input_length = X.shape[1]))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(176, dropout=dropout, recurrent_dropout=rec_dropout))
        model.add(tf.keras.layers.Dense(13, activation='softmax'))
        model.compile(loss = 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
        print(model.summary())

        # Splitting the data into training and testing data
        y=pd.get_dummies(self.df['sentiment'])
        X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)


        batch_size=64
        epochs = 30

        #for epoch in range(epochs):
        # Training the model using training data.
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose = 'auto')        
        accr = model.evaluate(X_test,y_test)      
        # Adding signature model
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))        
            
        # Saving local file               
        model.save("../models/my_model")       

        return model
    

    def predict(model,text):
        tokenizer = Tokenizer(num_words=500, split=' ') 
        tokenizer.fit_on_texts(text)
        seq = tokenizer.texts_to_sequences(text)
        X = pad_sequences(seq,23) # function is used to convert a list of sequences into a 2D NumPy array.
        pred = model.predict(X)
        print("pred",pred)
        labels = ['empty', 'sadness', 'enthusiasm' ,'neutral', 'worry' ,'surprise', 'love', 'fun',
                    'hate', 'happiness', 'boredom' ,'relief', 'anger']

        sentiment_class=labels[np.argmax(pred)]

        print('sentiment class',sentiment_class)

model = Emotions().modelling()
f = Emotions.predict(model,["I don not like pineapple on Pizza. I hate this idea"])