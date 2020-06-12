# Importing the libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Embedding, Bidirectional
from keras.models import load_model
import warnings
warnings.filterwarnings("ignore")



DATA_FOLDER_PATH = os.path.normpath(r'./data/')
MODEL_FOLDER_PATH = os.path.normpath(r'./models/')
# Normalizing paths
DATA_PATH = os.path.join(DATA_FOLDER_PATH, 'clean_train_data.csv')

# Importing dataset
data = pd.read_csv(DATA_PATH)

# Splitting the data
X = data['text'].values
y = data['target'].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
for train_ix, test_ix in skf.split(X, y):
    # select rows
    X_train, X_test = X[train_ix], X[test_ix]
    y_train, y_test = y[train_ix], y[test_ix]

# Vectorization
tokenizer = Tokenizer(num_words=20000, lower=True, split=' ')
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

X_train = pad_sequences(X_train, padding='pre')
X_test = pad_sequences(X_test, maxlen=34, padding='pre')
#X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
#X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)


# Model
model = Sequential()
model.add(Embedding(20000, 16, mask_zero=True))
model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(Dropout(0.2))
model.add(Bidirectional(LSTM(32, return_sequences=False)))
#model.add(Dropout(0.3))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
# print(model.summary())

# Uncomment the next line to inititate training process
history = model.fit(X_train, y_train, epochs=10,
                    batch_size=32, validation_data=(X_test, y_test))
model.save(os.path.join(MODEL_FOLDER_PATH, 'Basic-Model-Prototype-002.h5'))

print(model.evaluate(X_test, y_test))

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
