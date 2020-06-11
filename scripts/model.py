# Importing the libraries
import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Embedding, Bidirectional
import warnings
warnings.filterwarnings("ignore")


DATA_FOLDER_PATH = os.path.normpath(r'../data/')
## Normalizing paths
DATA_PATH = os.path.join(DATA_FOLDER_PATH, 'clean_train_data.csv')

# Importing dataset
data = pd.read_csv('../data/clean_train_data.csv')

# Splitting the data
X = data['text'].values
y = data['target'].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
for train_ix, test_ix in skf.split(X, y):
    # select rows
    X_train, X_test = X[train_ix], X[test_ix]
    y_train, y_test = y[train_ix], y[test_ix]

# Vectorization
tokenizer = Tokenizer(num_words=20000, lower=False, split=' ')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = pad_sequences(X_train)
X_test = pad_sequences(X_test, maxlen=34)

# Model
model = Sequential()
model.add(Embedding(20000, 64, input_length=34))
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(96, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(Dense(1, activation='softmax'))

model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
print(model.summary())
# Uncomment the next line to inititate training process
# model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))
