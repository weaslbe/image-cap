import io
import numpy as np
from keras.layers import Input, Concatenate, Dense
from keras.layers import Dropout, TimeDistributed, RepeatVector
from keras.layers import Embedding, LSTM, Multiply
from keras.models import Model


class LanguageModel:

    def __init__(self, dictionary_length,
                 sequence_length, lstm_cells,
                 predict_sequence, attention,
                 embedding_size, dense_size,
                 dense_layers, dropout_rate,
                 pre_build_embedding=False,
                 reverted_word_index=None):
        self.dictionary_length = dictionary_length
        self.sequence_length = sequence_length
        self.lstm_cells = lstm_cells
        self.predict_sequence = predict_sequence
        self.embedding_size = embedding_size
        self.dense_size = dense_size
        self.dropout_rate = dropout_rate
        self.dense_layers = dense_layers
        self.pre_build_embedding = pre_build_embedding
        self.reverted_word_index = reverted_word_index
        self.pretrained_word_vectors = self.load_vectors()

    def build_language_model(self, prev_words, encoder_output_shape):
        conv_feat = Input(shape=encoder_output_shape)

        conv_repeat = RepeatVector(self.sequence_length)(conv_feat)

        if self.pre_build_embedding:
            weights = self.load_embedding()
            emb = Embedding(self.dictionary_length,
                            self.embedding_size, weights=weights)
        else:
            emb = Embedding(self.dictionary_length, self.embedding_size)

        emb = emb(prev_words)

        lstm_in = Concatenate()([conv_repeat, emb])

        lstm = LSTM(self.lstm_cells,
                    return_sequences=self.predict_sequence)(lstm_in)

        if self.return_sequences:
            lstm = Concatenate()[conv_repeat, lstm]
        else:
            lstm = Concatenate()[conv_feat, lstm]

        if self.attention:
            attention_size = self.lstm_cells + encoder_output_shape[0]
            if self.return_sequences:
                attention = TimeDistributed(Dense(attention_size,
                                            activation='softmax'))(lstm)
            else:
                attention = Dense(attention_size,
                                  activation='softmax')(lstm)
            lstm = Multiply([attention, lstm])

        if self.predict_sequence:
            predictions = TimeDistributed(Dense(self.dictionary_length + 1,
                                                activation='softmax'),
                                          name='out')(lstm)
        else:
            for i in range(self.dense_layers):
                lstm = Dense(self.dense_size, activation='relu')(lstm)
                lstm = Dropout(self.dropout_rate)(lstm)
            predictions = Dense(self.dictionary_length + 1,
                                activation='softmax')(lstm)

        model = Model(input=[conv_feat, prev_words],
                      output=predictions)

        return model

    def load_embedding(self):
        weights = np.zeros(self.dictionary_length + 1, self.embedding_size)
        #for word_index in range(self.dictionary_length):
        for word_index in range(self.dictionary_length[:10]):
            word = self.reverted_word_index[word_index + 1]
            emb = self.get_embedding(word)
            weights[word_index + 1] = emb
        return weights

    def load_vectors(self, fname="data/embeddings/wiki-news-300d-1M.vec"):
        fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        n, d = map(int, fin.readline().split())
        data = {}
        for line in fin:
            tokens = line.rstrip().split(' ')
            data[tokens[0]] = map(float, tokens[1:])
        return data

    # input: a word
    # output: fastText word-embedding
    def get_embedding(self, word):
        return self.pretrained_word_vectors[word]
