# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 12:45:55 2019

@author: Ishita
"""

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score,classification_report, precision_recall_fscore_support
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from kutilities.callbacks import MetricsCallback, PlottingCallback
import numpy as np
from keras.layers import Input, Dense, Embedding, Concatenate, Activation, \
    Dropout, LSTM, Bidirectional, GlobalMaxPooling1D, GaussianNoise
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from kutilities.callbacks import MetricsCallback, PlottingCallback
from sklearn.metrics import f1_score, precision_score, recall_score
from keras.callbacks import ModelCheckpoint, TensorBoard
import re
import io

label2emotion = {0: "others", 1: "happy", 2: "sad", 3: "angry"}
emotion2label = {"others": 0, "happy": 1, "sad": 2, "angry": 3}

emoticons_additional = {
    '(^・^)': '<happy>', ':‑c': '<sad>', '=‑d': '<happy>', ":'‑)": '<happy>', ':‑d': '<laugh>',
    ':‑(': '<sad>', ';‑)': '<happy>', ':‑)': '<happy>', ':\\/': '<sad>', 'd=<': '<annoyed>',
    ':‑/': '<annoyed>', ';‑]': '<happy>', '(^�^)': '<happy>', 'angru': 'angry', "d‑':":
        '<annoyed>', ":'‑(": '<sad>', ":‑[": '<annoyed>', '(�?�)': '<happy>', 'x‑d': '<laugh>',
}

text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
               'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
              'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens
    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter",
    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter",
    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words
    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,
    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionaries.
    dicts=[emoticons, emoticons_additional]
)


def tokenize(text):
    print("in tokenize")
    text = " ".join(text_processor.pre_process_doc(text))
    return text


def preprocessData(dataFilePath, mode):
    print("in preprocess data")
    conversations = []
    labels = []
    with io.open(dataFilePath, encoding="utf8") as finput:
        finput.readline()
        for line in finput:
            line = line.strip().split('\t')
            for i in range(1, 4):
                line[i] = tokenize(line[i])
            if mode == "train":
                labels.append(emotion2label[line[4]])
            conv = line[1:4]
            conversations.append(conv)
    if mode == "train":
        return np.array(conversations), np.array(labels)
    else:
        return np.array(conversations)
    
texts_train, labels_train = preprocessData('C:/Users/Ishita Jain/Desktop/semeval19_emocon/train.txt', mode="train")
texts_dev, labels_dev = preprocessData('C:/Users/Ishita Jain/Desktop/semeval19_emocon/valid.txt', mode="train")
texts_test, labels_test = preprocessData('C:/Users/Ishita Jain/Desktop/semeval19_emocon/test.txt', mode="train") 

def getEmbeddings(file):
    print("in getEmbeddings")
    embeddingsIndex = {}
    dim = 0
    with io.open(file, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            embeddingVector = np.asarray(values[1:], dtype='float32')
            embeddingsIndex[word] = embeddingVector 
            dim = len(embeddingVector)
    return embeddingsIndex, dim


def getEmbeddingMatrix(wordIndex, embeddings, dim):
    print("in getEmbeddingsmatrix")
    embeddingMatrix = np.zeros((len(wordIndex) + 1, dim))
    for word, i in wordIndex.items():
        embeddingMatrix[i] = embeddings.get(word)
    return embeddingMatrix  



#embeddings, dim = getEmbeddings('C:/Users/Ishita Jain/Desktop/emosense.300d.txt/emosense.300d.txt')
#tokenizer = Tokenizer(filters='')
#tokenizer.fit_on_texts([' '.join(list(embeddings.keys()))])
#
#wordIndex = tokenizer.word_index
#print("Found %s unique tokens." % len(wordIndex))
#
#embeddings_matrix = getEmbeddingMatrix(wordIndex, embeddings, dim)
#
#
#
#MAX_SEQUENCE_LENGTH = 24
#
#X_train, X_val, y_train, y_val = train_test_split(texts_train, labels_train, test_size=0.2, random_state=42)
#
#labels_categorical_train = to_categorical(np.asarray(y_train))
#labels_categorical_val = to_categorical(np.asarray(y_val))
#labels_categorical_dev = to_categorical(np.asarray(labels_dev))
#labels_categorical_test = to_categorical(np.asarray(labels_test))


def get_sequances(texts, sequence_length):
    print("in get_sequances")
    message_first = pad_sequences(tokenizer.texts_to_sequences(texts[:, 0]), sequence_length)
    message_second = pad_sequences(tokenizer.texts_to_sequences(texts[:, 1]), sequence_length)
    message_third = pad_sequences(tokenizer.texts_to_sequences(texts[:, 2]), sequence_length)
    return message_first, message_second, message_third


message_first_message_train, message_second_message_train, message_third_message_train = get_sequances(X_train, MAX_SEQUENCE_LENGTH)
message_first_message_val, message_second_message_val, message_third_message_val = get_sequances(X_val, MAX_SEQUENCE_LENGTH)
message_first_message_dev, message_second_message_dev, message_third_message_dev = get_sequances(texts_dev, MAX_SEQUENCE_LENGTH)
message_first_message_test, message_second_message_test, message_third_message_test = get_sequances(texts_test, MAX_SEQUENCE_LENGTH)




def buildModel(embeddings_matrix, sequence_length, lstm_dim, hidden_layer_dim, num_classes, 
               noise=0.1, dropout_lstm=0.2, dropout=0.2):
    print("build model")
    turn1_input = Input(shape=(sequence_length,), dtype='int32')
    turn2_input = Input(shape=(sequence_length,), dtype='int32')
    turn3_input = Input(shape=(sequence_length,), dtype='int32')
    embedding_dim = embeddings_matrix.shape[1]
    embeddingLayer = Embedding(embeddings_matrix.shape[0],
                                embedding_dim,
                                weights=[embeddings_matrix],
                                input_length=sequence_length,
                                trainable=False)
    
    turn1_branch = embeddingLayer(turn1_input)
    turn2_branch = embeddingLayer(turn2_input) 
    turn3_branch = embeddingLayer(turn3_input) 
    
    turn1_branch = GaussianNoise(noise, input_shape=(None, sequence_length, embedding_dim))(turn1_branch)
    turn2_branch = GaussianNoise(noise, input_shape=(None, sequence_length, embedding_dim))(turn2_branch)
    turn3_branch = GaussianNoise(noise, input_shape=(None, sequence_length, embedding_dim))(turn3_branch)

    lstm1 = Bidirectional(LSTM(lstm_dim, dropout=dropout_lstm))
    lstm2 = Bidirectional(LSTM(lstm_dim, dropout=dropout_lstm))
    
    turn1_branch = lstm1(turn1_branch)
    turn2_branch = lstm2(turn2_branch)
    turn3_branch = lstm1(turn3_branch)
    
    x = Concatenate(axis=-1)([turn1_branch, turn2_branch, turn3_branch])
    
    x = Dropout(dropout)(x)
    
    x = Dense(hidden_layer_dim, activation='relu')(x)
    
    output = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=[turn1_input, turn2_input, turn3_input], outputs=output)
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    
    return model

#model = buildModel(embeddings_matrix, MAX_SEQUENCE_LENGTH, lstm_dim=64, hidden_layer_dim=30, num_classes=4)
#print("model summary")
#model.summary()
#
#
#
#metrics = {
#    "f1_e": (lambda y_test, y_pred:
#             f1_score(y_test, y_pred, average='micro',
#                      labels=[emotion2label['happy'],
#                              emotion2label['sad'],
#                              emotion2label['angry']
#                              ])),
#    "precision_e": (lambda y_test, y_pred:
#                    precision_score(y_test, y_pred, average='micro',
#                                    labels=[emotion2label['happy'],
#                                            emotion2label['sad'],
#                                            emotion2label['angry']
#                                            ])),
#    "recoll_e": (lambda y_test, y_pred:
#                 recall_score(y_test, y_pred, average='micro',
#                              labels=[emotion2label['happy'],
#                                      emotion2label['sad'],
#                                      emotion2label['angry']
#                                      ]))
#}
#
#_datasets = {}
#_datasets["dev"] = [[message_first_message_dev, message_second_message_dev, message_third_message_dev],
#                    np.array(labels_categorical_dev)]
#_datasets["val"] = [[message_first_message_val, message_second_message_val, message_third_message_val],
#                    np.array(labels_categorical_val)]
#
#metrics_callback = MetricsCallback(datasets=_datasets, metrics=metrics)
#
#filepath = "C:/Users/Ishita Jain/Desktop/bidirectional_LSTM_best_weights_{epoch:02d}-{val_acc:.4f}.hdf5"
#checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=True, save_weights_only=False,
#                             mode='auto', period=1)
#tensorboardCallback = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)
#
#history = model.fit([message_first_message_train, message_second_message_train, message_third_message_train],
#                    np.array(labels_categorical_train),
#                    callbacks=[metrics_callback, checkpoint, tensorboardCallback],
#                    validation_data=(
#                        [message_first_message_val, message_second_message_val, message_third_message_val],
#                        np.array(labels_categorical_val)
#                    ),
#                    epochs=5,
#                    batch_size=10)
                   
model.load_weights("C:/Users/Ishita Jain/Desktop/bidirectional_LSTM_best_weights_03-0.9110.hdf5")                    
y_pred = model.predict([message_first_message_dev, message_second_message_dev, message_third_message_dev])
from sklearn.metrics import classification_report

for title, metric in metrics.items():
    print(title, metric(labels_categorical_dev.argmax(axis=1), y_pred.argmax(axis=1)))
print(classification_report(labels_categorical_dev.argmax(axis=1), y_pred.argmax(axis=1)))
                  