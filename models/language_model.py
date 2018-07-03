import io
import numpy as np
from keras.layers import Input, Concatenate, Dense
from keras.layers import Dropout, TimeDistributed, RepeatVector
from keras.layers import Embedding, CuDNNLSTM, Multiply
from keras.models import Model
from fastText import load_model


class LanguageModel:

    def __init__(self, dictionary_length,
                 sequence_length, lstm_cells=1024,
                 predict_sequence=True, attention=False,
                 embedding_size=300, dense_size=0,
                 dense_layers=0, dropout_rate=0.0,
                 pre_build_embedding=False,
                 reverted_word_index=None,
                 fast_text_model=None):
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
        self.fast_text_model = fast_text_model

    def build_language_model(self, prev_words, conv_feat,
                             encoder_output_shape, img_input):
#        conv_feat = Input(shape=encoder_output_shape)

#        conv_repeat = RepeatVector(self.sequence_length)(conv_feat)

        if self.pre_build_embedding:
            weights = self.load_embedding()
            emb = Embedding(self.dictionary_length,
                            self.embedding_size, weights=weights)
        else:
            emb = Embedding(self.dictionary_length, self.embedding_size)

        emb = emb(prev_words)

#        lstm_in = Concatenate()([conv_repeat, emb])

        lstm = CuDNNLSTM(encoder_output_shape[0],
                         return_sequences=self.predict_sequence)

        lstm = lstm(emb, initial_state=[conv_feat, conv_feat])

#        if self.return_sequences:
#            lstm = Concatenate()[conv_repeat, lstm]
#        else:
#            lstm = Concatenate()[conv_feat, lstm]

#        if self.attention:
#            attention_size = self.lstm_cells + encoder_output_shape[0]
##            if self.return_sequences:
#                attention = TimeDistributed(Dense(attention_size,
#                                            activation='softmax'))(lstm)
#            else:
#                attention = Dense(attention_size,
#                                  activation='softmax')(lstm)
#            lstm = Multiply([attention, lstm])

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

        model = Model(input=[img_input, prev_words],
                      output=predictions)

        return model

    def load_embedding(self):
        weights = np.zeros(self.dictionary_length + 1, self.embedding_size)
        with load_model(self.fast_text_model) as f_model:
            for word_index in range(self.dictionary_length):
                word = self.reverted_word_index[word_index + 1]
                emb = f_model.get_word_vector(word)
                weights[word_index + 1] = emb
        return weights
