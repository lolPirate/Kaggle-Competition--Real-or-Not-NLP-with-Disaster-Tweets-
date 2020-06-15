import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import numpy as np
import re
import unicodedata
import string



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

    def remove_social_words(self, data):
        p = re.compile('[@,#]\w+')
        return re.sub(p, ' ', data.strip())

    def remove_punctuations(self, data):
        p = str.maketrans("","",string.punctuation)
        #p = str.maketrans(string.punctuation, ' '*len(string.punctuation))
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
    
    def convert_lower(self, data):
        return data.lower()

    def remove_stopwords(self, data):
        sw = set(stopwords.words('english'))
        word_tokens = word_tokenize(data)
        filtered_sentence = [w for w in word_tokens if not w in sw]
        return " ".join(filtered_sentence)
    
    def remove_numbers(self, data):
        p = str.maketrans("","",'0123456789')
        return data.translate(p)

    def apply_lemmatization(self, data):
        tokens = word_tokenize(data)
        return " ".join([WordNetLemmatizer().lemmatize(i) for i in tokens])
    
    def remove_nouns(self, data):
        tokens = word_tokenize(data)
        return " ".join([text for text, tag in nltk.tag.pos_tag(tokens) if not tag in ['NNP','NNPS']])


    def clean_data(self, data):
        data['text'] = data['text'].apply(lambda x: self.remove_html(x))
        data['text'] = data['text'].apply(lambda x: self.remove_url(x))
        data['text'] = data['text'].apply(lambda x: self.remove_emojis(x))
        data['text'] = data['text'].apply(lambda x: self.remove_social_words(x))
        data['text'] = data['text'].apply(lambda x: self.remove_punctuations(x))
        data['text'] = data['text'].apply(lambda x: self.remove_control_characters(x))
        extra_chars = "".join(self.char_counter(data['text'])[64:])
        data['text'] = data['text'].apply(lambda x: self.remove_unknown_characters(x, extra_chars))
        data['text'] = data['text'].apply(lambda x: self.remove_extra_spaces(x))
        data['text'] = data['text'].apply(lambda x: self.convert_lower(x))
        data['text'] = data['text'].apply(lambda x: self.remove_stopwords(x))
        data['text'] = data['text'].apply(lambda x: self.remove_numbers(x))
        data['text'] = data['text'].apply(lambda x: self.apply_lemmatization(x))
        data['text'] = data['text'].apply(lambda x: self.remove_nouns(x))
        data = data[['text', 'target']]
        data = data.drop_duplicates()
        return data

class Disaster_Prediction_Model():

    def create_model(self):
        model = Sequential()
        model.add(Embedding(12802, 64, mask_zero=True))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(128, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(64, return_sequences=True)))
        model.add(Dropout(0.2))
        model.add(Bidirectional(LSTM(64, return_sequences=False)))
        model.add(Dropout(0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

class Agent():

    def __init__(self):
        self.model = Disaster_Prediction_Model()
        self.model = self.model.create_model()
    
    def preprocess_data(self, data):
        x = data['text'].values
        y = data['target'].values

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, stratify=y, shuffle=True)

        tokenizer = Tokenizer(num_words=12802, split=' ')
        tokenizer.fit_on_texts(x_train)
        x_train = tokenizer.texts_to_sequences(x_train)
        x_test = tokenizer.texts_to_sequences(x_test)

        x_train = pad_sequences(x_train, padding='post')
        x_test = pad_sequences(x_test, padding='post')


        df = pd.DataFrame.from_dict(zip(x_train, y_train))
        df.to_csv('op.csv', index=False)

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
    #agent.save_model()
