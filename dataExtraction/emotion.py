# Importing Required librarues
from fileinput import filename
from select import select
import pandas as pd
import tensorflow as tf
import numpy as np
import string
import random
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse
import data_cleaning
import keras_tuner as kt
from pprint import pprint
from sklearn.metrics import accuracy_score
import warnings 
warnings.filterwarnings('ignore')
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

# Class to predicts emotions of the tweets
class Emotions():

    def __init__(self):
        '''train.txt includes 13 diffrent lables'''
        data = pd.read_csv('train.txt',sep=',',header=0)
        self.df = pd.DataFrame(data)


    def reverser(self,inp,inv_word_index):
        st = ''
        for i in inp:
            st += ' ' + str(inv_word_index.get(i))
        return st


    def predict(self,x_test,y_test,inv_word_index,inv_label_index,model):
        '''Predictinf the emotions for each tweet'''
        seed = random.randint(0, x_test.shape[0])
        x_seed = x_test[seed]
        act = y_test[seed]
        st = self.reverser(x_seed,inv_word_index)
        print('Input_Sentence: ', st)
        print('--'*20)
        print('Actual_emotion: ', inv_label_index.get(act))
        pred = (np.argmax(model.predict(x_seed.reshape(1,-1)), axis=-1).tolist())
        print('--'*20)
        print('Predicted_emotion: ', inv_label_index.get(pred[-1]))  

  
    def test_preprocess(self,test_data, token_data, label_index):
        
        content_text = test_data.content.tolist()
        all_classes = test_data.sentiment.unique().tolist()

        content_text = test_data.content.tolist()
        ex_char = string.punctuation
        ex_char = ex_char.replace('~', '')
        c_text = '~~~~~~'.join(content_text)

        x = c_text.translate(str.maketrans('', '', ex_char))
        c_text = x.split('~~~~~~')

        print('Again_Test_samples: ',len(c_text))
        print('Some_Test_Sentences: ')
        print()
        print(c_text[34])
        print(c_text[21])
        print()

        ind_text = token_data.texts_to_sequences(c_text)
        x_test = np.array(ind_text)
        print()
        print('All_Test_samples: ', len(ind_text))
        y_test = []
        for i in test_data.sentiment:
            y_test.append(label_index.get(i))
        y_test = np.array(y_test)
        print('Label_shape: ', y_test.shape)
        
        return x_test, y_test        


    def clean_text(self,train_data,test_data):  
        '''Using our datacleaning class'''    
        cleaningClass = data_cleaning.DataCleaning()     
        train_data['content'] = train_data[['content']].applymap(
            lambda x: cleaningClass.data_preprocessing(x))
      
        content_text = train_data.content.tolist()

        token_data = tf.keras.preprocessing.text.Tokenizer(num_words=None,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' ')
        token_data.fit_on_texts(content_text)
        ind_text = token_data.texts_to_sequences(content_text)
        word_index = token_data.word_index
        all_classes = train_data.sentiment.unique().tolist()
        label_token = tf.keras.preprocessing.text.Tokenizer(num_words=len(all_classes),
        filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
        lower=True,
        split=' ')

        classes = ' '.join(all_classes)
        label_token.fit_on_texts([classes])
        label_index = label_token.word_index
        print('No. of Labels: ',len(list(label_index)))
        print()     
        label_index = {key:value-1 for key, value in label_index.items()}
        pprint(label_index)
        inv_label_index = {value:key for key, value in label_index.items()}
        print()
        pprint(inv_label_index)

        x_train = np.array(ind_text)
        max_inp_len = len(x_train[0])
        for _, i in enumerate(x_train):
            if len(i) > max_inp_len:
                max_inp_len = len(i)
        print('max_input_length: ',max_inp_len)

        x_train = tf.keras.preprocessing.sequence.pad_sequences(x_train, maxlen = max_inp_len, padding = 'post')
        
        print('x_train_shape: ',x_train.shape, ', y_train_shape: ', train_data['sentiment'].shape)


        print('Test_Preprocessing ...')
        x_test, y_test = self.test_preprocess(test_data, token_data, label_index)
        print(x_test.shape, y_test.shape)

        x_test = tf.keras.preprocessing.sequence.pad_sequences(x_test, maxlen = max_inp_len, padding = 'post')
        print('x_test_shape: ',x_test.shape, ', y_test_shape: ', y_test.shape)
        ''' Normal_RNN_Model '''

        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience = 2)
            
        input_1 = tf.keras.layers.Input(shape=(x_train.shape[-1],))
        embd_1 = tf.keras.layers.Embedding(input_dim = len(list(word_index)) + 1, output_dim=128)(input_1)
        units = 64
        dropout=0.3
        recurrent_dropout=0.5
        bi_1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units= units, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout), merge_mode = 'concat')(embd_1)
        lstm_1 = tf.keras.layers.LSTM(units=units, dropout=dropout, recurrent_dropout=recurrent_dropout)(bi_1)
        dense_1 = tf.keras.layers.Dense(len(all_classes), activation='softmax')(lstm_1)

        '''Three important lines that is missed in the MLFlow Documentation'''

        # (1) update the data in the running real time
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

        # (2) this is model registry
        registery_uri = 'sqlite:///mlflow.db'

        # (3) update the date in the real time from the above sqlitedb
        mlflow.tracking.set_tracking_uri(registery_uri)
        
        # Running the MLflow
        with mlflow.start_run():
            model = tf.keras.models.Model(inputs = input_1, outputs = dense_1, name='Basic_LSTM')
            model.compile(loss='sparse_categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
            mlflow.log_param("units", units)
            mlflow.log_param("dropout", dropout)
            mlflow.log_param("recurrent_dropout", recurrent_dropout)
              
            tf.keras.utils.plot_model(model, show_shapes =True)
            print()
            print(' -- Model_Evaluation -- ')
            model_val = model.evaluate(x_test, y_test)
            print(model_val)

            index = {}
            for val, key in enumerate(word_index.keys()):
                index[val+1] = key 
            inv_word_index = index
            inv_word_index[0] = ''
 

            for _ in range(5):
                print('**'*50)
                self.predict(x_test,y_test,inv_word_index,inv_label_index,model)
                mlflow.log_metric("accuracy",model_val[1] )
                mlflow.log_metric("step - loss",model_val[0])

                print('**'*50)
                print()
            
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Model registry does not work with file store
            if tracking_url_type_store != "file":

            # Register the model
            # There are other ways to use the Model Registry, which depends on the use case,
            # please refer to the doc for more information:
            # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                mlflow.sklearn.log_model(model, "model", registered_model_name="EmotionsModel")
            else:
                mlflow.sklearn.log_model(model, "model")    


    def preprocess(self):
        train_data,test_data = train_test_split(self.df, train_size = 0.7, random_state=42)
        
        maxx = 3
        all_classes = train_data['sentiment'].unique().tolist()
        target_majority = train_data[train_data.sentiment==all_classes[maxx]]

        for cl in range(13):
            train_minority = self.df[self.df.sentiment==all_classes[cl]]
            train_minority_upsampled = resample(train_minority, replace=True, n_samples=len(target_majority), random_state=123)
            if cl == 0:
                train_upsampled = pd.concat([train_minority_upsampled, target_majority])
            if cl>0 and cl!=maxx:
                train_upsampled = pd.concat([train_minority_upsampled, train_upsampled])

        self.clean_text(train_upsampled,test_data)


e = Emotions()

e.preprocess()