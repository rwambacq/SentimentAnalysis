# Imports
# -------

import gc

import numpy
import pandas

from keras.layers import Dense, Embedding, LSTM, Conv1D, MaxPooling1D, Dropout, GRU, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split

from nltk.tokenize import word_tokenize
from nltk import FreqDist
from nltk.stem import WordNetLemmatizer

import re

import sys


# orig_stdout = sys.stdout
# f = open('SentimentAnalysisOutput3', 'w')
# sys.stdout = f

# Opkuis functie definiëren
# -------------------------

def opkuisen(kolom):
    wnl = WordNetLemmatizer()
    toReturn = []
    for x in kolom:
        lijn = str(x)
        lijn = re.sub('[^a-zA-Z]', ' ', lijn)

        lijn_lemmatized = []
        for woord in re.findall(r"[\w']+|[.,!?;]", lijn.lower()):
            lijn_lemmatized.append(wnl.lemmatize(woord))
        lijn = ' '.join(lijn_lemmatized)
        toReturn.append(lijn)
    return toReturn

def runnen(modelvar, modelnaam):
    model = modelvar
    model_naam = modelnaam

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

    fitten = model.fit(model_train, sentimenten_train, validation_data=(model_validate, sentimenten_validate), epochs=5, batch_size=128, verbose=1)

    prediction = model.predict_classes(model_test)

    sample.Sentiment = prediction
    sample.to_csv('./outputs/' + model_naam + '_output.csv', index=False)

# Preprocessing
# -------------

gc.collect()

train = pandas.read_csv('./files/train.tsv', sep='\t')
test = pandas.read_csv('./files/test.tsv', sep='\t')
sample = pandas.read_csv('./files/sampleSubmission.csv')

test["Sentiment"] = -1

temp = pandas.concat([train, test], ignore_index=True)
del train, test
gc.collect()

temp['opgekuist'] = opkuisen(temp.Phrase.values)

test = temp[temp.Sentiment == -1]
train = temp[temp.Sentiment != -1]
test.drop('Sentiment', axis=1, inplace=True)

del temp
gc.collect()

sentimenten = train.Sentiment.values
cat_sentimenten = to_categorical(sentimenten)

traintekst = train.opgekuist.values
testtekst = test.opgekuist.values

train_kleiner, train_validate, sentimenten_train, sentimenten_validate = train_test_split(traintekst, cat_sentimenten, test_size=0.2, stratify=cat_sentimenten, random_state=42)

s = word_tokenize(' '.join(train_kleiner))
aantalwoorden = len(set(s))

maxlengte = 0
for t in train_kleiner:
    tok = word_tokenize(t)
    if len(tok) > maxlengte:
        maxlengte = len(tok)

tz = Tokenizer(num_words=aantalwoorden)
tz.fit_on_texts(list(train_kleiner))

model_train = tz.texts_to_sequences(train_kleiner)
model_train = sequence.pad_sequences(model_train, maxlen=maxlengte)
model_test = tz.texts_to_sequences(testtekst)
model_test = sequence.pad_sequences(model_test, maxlen=maxlengte)

model_validate = tz.texts_to_sequences(train_validate)
model_validate = sequence.pad_sequences(model_validate, maxlen=maxlengte)


# LSTM
# ----

# 1 Layer

LSTM1model = Sequential([
        Embedding(aantalwoorden, 250, mask_zero=True),
        LSTM(128, dropout=0.4, recurrent_dropout=0.4, return_sequences=False),
        Dense(5, activation='softmax')
    ])

#runnen(LSTM1model, "LSTM1model")

# 2 Layers

LSTM2model = Sequential([
    Embedding(aantalwoorden, 250, mask_zero=True),
    LSTM(128, dropout=0.4, recurrent_dropout=0.4, return_sequences=True),
    LSTM(64, dropout=0.5, recurrent_dropout=0.5, return_sequences=False),
    Dense(5, activation='softmax')
])

#runnen(LSTM2model, "LSTM2opt")

# 3 Layers

LSTM3model = Sequential([
    Embedding(aantalwoorden, 250, mask_zero=True),
    LSTM(128, dropout=0.4, recurrent_dropout=0.4, return_sequences=True),
    LSTM(64, dropout=0.5, recurrent_dropout=0.5, return_sequences=True),
    LSTM(32, dropout=0.6, recurrent_dropout=0.6, return_sequences=False),
    Dense(5, activation='softmax')
])

#runnen(LSTM3model, "LSTM3model")

# GRU
# ---

# 1 Layer

GRU1model = Sequential([
    Embedding(aantalwoorden, 250, mask_zero=True),
    GRU(128),
    Dense(5, activation='softmax')
])

#runnen(GRU1model, "GRU1model")

# 2 Layers

GRU2model = Sequential([
    Embedding(aantalwoorden, 250, mask_zero=True),
    GRU(128, return_sequences=True),
    GRU(64, return_sequences=False),
    Dense(5, activation='softmax')
])

#runnen(GRU2model, "GRU2model")

# 3 Layers

GRU3model = Sequential([
    Embedding(aantalwoorden, 250, mask_zero=True),
    GRU(128, return_sequences=True),
    GRU(64, return_sequences=True),
    GRU(32, return_sequences=False),
    Dense(5, activation='softmax')
])

#runnen(GRU3model, "GRU3model")

# CNN
# ---

#  1 Layer (Conv1D & MaxPooling1D)

CNN1model = Sequential([
Embedding(aantalwoorden, 100, input_length=maxlengte),
Conv1D(64, kernel_size=3, activation='relu'),
MaxPooling1D(pool_size=2),
Flatten(),
Dense(128, activation='relu'),
Dropout(0.5),
Dense(5, activation='softmax')
])

#runnen(CNN1model, "CNN1model")

# 2 Layers (Conv1D & MaxPooling1D)

CNN2model = Sequential([
Embedding(aantalwoorden, 100, input_length=maxlengte),
Conv1D(32, kernel_size=3, activation='relu'),
MaxPooling1D(pool_size=2),
Conv1D(64, kernel_size=3, activation='relu'),
MaxPooling1D(pool_size=2),
Flatten(),
Dense(128, activation='relu'),
Dropout(0.5),
Dense(5, activation='softmax')
])

#runnen(CNN2model, "CNN2model")

# 3 Layers (Conv1D & MaxPooling1D)

CNN3model = Sequential([
Embedding(aantalwoorden, 100, input_length=maxlengte),
Conv1D(32, kernel_size=3, activation='relu'),
MaxPooling1D(pool_size=2),
Conv1D(64, kernel_size=3, activation='relu'),
MaxPooling1D(pool_size=2),
Conv1D(128, kernel_size=3, activation='relu'),
MaxPooling1D(pool_size=2),
Flatten(),
Dense(128, activation='relu'),
Dropout(0.5),
Dense(5, activation='softmax')
])

#runnen(CNN3model, "CNN3model")

# CNN + GRU/LSTM
# --------------

# Eerst CNN, dan GRU

CNNGRUmodel = Sequential([
Embedding(aantalwoorden, 100, input_length=maxlengte),
Conv1D(64, kernel_size=3, padding='same', activation='relu'),
MaxPooling1D(pool_size=2),
Dropout(0.25),
GRU(128, return_sequences=True),
Dropout(0.3),
Flatten(),
Dense(128, activation='relu'),
Dropout(0.5),
Dense(5, activation='softmax')
])

#runnen(CNNGRUmodel, "CNNGRUmodel")

# Eerst GRU, dan CNN

GRUCNNmodel = Sequential([
Embedding(aantalwoorden, 100, input_length=maxlengte),
GRU(128, return_sequences=True),
Dropout(0.3),
Conv1D(64, kernel_size=3, padding='same', activation='relu'),
MaxPooling1D(pool_size=2),
Flatten(),
Dense(128, activation='relu'),
Dropout(0.5),
Dense(5, activation='softmax')
])

#runnen(GRUCNNmodel, "GRUCNNmodel")

# Eerst CNN, dan LSTM

CNNLSTMmodel = Sequential([
Embedding(aantalwoorden, 100, input_length=maxlengte),
Conv1D(64, kernel_size=3, padding='same', activation='relu'),
MaxPooling1D(pool_size=2),
Dropout(0.25),
LSTM(128, return_sequences=False),
Dropout(0.3),
Dense(128, activation='relu'),
Dropout(0.5),
Dense(5, activation='softmax')
])

#runnen(CNNLSTMmodel, "CNNLSTMmodel")

# Eerst LSTM, dan CNN

LSTMCNNmodel = Sequential([
Embedding(aantalwoorden, 100, input_length=maxlengte),
LSTM(128, return_sequences=True),
Dropout(0.3),
Conv1D(64, kernel_size=3, padding='same', activation='relu'),
MaxPooling1D(pool_size=2),
Flatten(),
Dense(128, activation='relu'),
Dropout(0.5),
Dense(5, activation='softmax')
])

#runnen(LSTMCNNmodel, "LSTMCNNmodel")

# sys.stdout = orig_stdout
# f.close()
