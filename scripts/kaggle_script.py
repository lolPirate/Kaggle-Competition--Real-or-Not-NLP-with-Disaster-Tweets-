import os
import string
import unicodedata
import re
import numpy as np
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional
from keras.models import load_model



class Data_Loader():
    def __init__(self, data_folder_path):
        self.DATA_FOLDER_PATH = os.path.normpath(data_folder_path)

    def load_data(self, file):
        df = pd.read_csv(os.path.join(self.DATA_FOLDER_PATH, file+'.csv'))
        return df

    def remove_html(self, data):
        p = re.compile(r'<.*?>')
        return p.sub(' ', data)

    def remove_url(self, data):
        p = re.compile(r"http\S+")
        return re.sub(p, 'link', data)

    def remove_emojis(self, data):
        p = re.compile("["
                       u"\U0001F600-\U0001F64F"  # emoticons
                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       "]+", flags=re.UNICODE)
        return(p.sub(r' ', data))

    def remove_punctuations(self, data):
        p = str.maketrans(string.punctuation, ' '*len(string.punctuation))
        return data.translate(p)

    def remove_control_characters(self, data):
        return "".join(ch for ch in data if unicodedata.category(ch)[0] != "C")

    def char_counter(self, data):
        char_count = Counter()
        for i in list(data):
            for c in i:
                char_count[c] += 1
        return sorted(char_count.keys())

    def remove_unknown_characters(self, data, extra_characters):
        p = str.maketrans(extra_characters, ' '*len(extra_characters))
        return data.translate(p)

    def remove_extra_spaces(self, data):
        p = re.compile('\s\s+')
        return re.sub(p, ' ', data.strip())

    def clean_data(self, data):
        data['text'] = data['text'].apply(lambda x: self.remove_html(x))
        data['text'] = data['text'].apply(lambda x: self.remove_url(x))
        data['text'] = data['text'].apply(lambda x: self.remove_emojis(x))
        data['text'] = data['text'].apply(lambda x: self.remove_punctuations(x))
        data['text'] = data['text'].apply(lambda x: self.remove_control_characters(x))
        extra_chars = "".join(self.char_counter(data['text'])[63:])
        data['text'] = data['text'].apply(lambda x: self.remove_unknown_characters(x, extra_chars))
        data['text'] = data['text'].apply(lambda x: self.remove_extra_spaces(x))
        data = data[['text', 'target']]
        data.drop_duplicates(inplace=True)

        return data

class Disaster_Prediction_Model():
    def create_model(self):
        model = Sequential()
        model.add(Embedding(20000, 16, mask_zero=True))
        model.add(Bidirectional(LSTM(32, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(32, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(32, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(32, return_sequences=False)))
        model.add(Dropout(0.3))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return(model)

class Agent():

    def __init__(self):
        self.model = Disaster_Prediction_Model()
        self.model = self.model.create_model()
    
    def preprocess_data(self, data):
        x = data['text'].values
        y = data['target'].values

        x_train, x_test, y_train, y_test = [], [], [], []

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        for train_ix, test_ix in skf.split(x, y):
            x_train, x_test = x[train_ix], x[test_ix]
            y_train, y_test = y[train_ix], y[test_ix]

        tokenizer = Tokenizer(num_words=20000, lower=True, split=' ')
        tokenizer.fit_on_texts(x_train)
        x_train = tokenizer.texts_to_sequences(x_train)
        x_test = tokenizer.texts_to_sequences(x_test)

        x_train = pad_sequences(x_train, padding='pre')
        x_test = pad_sequences(x_test, maxlen=34, padding='pre')
        #x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
        #x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)

        return x_train, y_train, x_test, y_test

    def train(self, x_train, y_train, x_test, y_test):
        self.model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

    def save_model(self):
        self.model.save('Basic-Model.h5')


if __name__ == '__main__':
    data_folder_path = r'./data/'
    data_loader = Data_Loader(data_folder_path)
    train_data = data_loader.load_data('train')
    train_data = data_loader.clean_data(train_data)
    
    agent = Agent()
    agent.train(*agent.preprocess_data(train_data))
    agent.save_model()
