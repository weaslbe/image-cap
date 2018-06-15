from keras.layers import Conv2D, Input, Concatenate, Dense, Dropout, TimeDistributed, RepeatVector, Embedding, Flatten, LSTM
from keras.models import Model


class LanguageModel:

    def __init__(self, dictionary_length, sequence_length):
        self.dictionary_length = dictionary_length
        self.sequence_length = sequence_length

    def build_language_model(self, prev_words, encoder_output_shape):
        conv_feat = Input(shape=encoder_output_shape)

        conv_repeat = RepeatVector(self.sequence_length)(conv_feat)

        emb = Embedding(self.dictionary_length, 300)(prev_words)

        lstm_in = Concatenate()([conv_repeat, emb])

        lstm = LSTM(128, return_sequences=True)(lstm_in)

        predictions = TimeDistributed(Dense(self.dictionary_length + 1,
                                            activation='softmax'),
                                      name='out')(lstm)

        model = Model(input=[conv_feat, prev_words],
                      output=predictions)

        return model
