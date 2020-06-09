# Importing the libraries
import os
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings("ignore")


DATA_FOLDER_PATH = os.path.normpath(r'../data/')
## Normalizing paths
DATA_PATH = os.path.join(DATA_FOLDER_PATH, 'clean_train_data.csv')

# Importing dataset
data = pd.read_csv('../data/clean_train_data.csv')
print(data.head())
