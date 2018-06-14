from keras.layers import Conv2D, Input, Concatenate, Dense, Dropout, TimeDistributed, RepeatVector, Embedding, Flatten, LSTM
from keras.models import Model


class QuickImageCaptioningModel:

    def build_image_model(self, coco_image):
        conv_blocks = []
        conv_1 = Conv2D(5, (3, 3), activation='relu', data_format='channels_last')(coco_image)
        conv_2 = Conv2D(5, (5, 5), activation='relu', data_format='channels_last')(coco_image)
        conv_3 = Conv2D(5, (7, 7), activation='relu', data_format='channels_last')(coco_image)
        conv_1 = Flatten()(conv_1)
        conv_2 = Flatten()(conv_2)
        conv_3 = Flatten()(conv_3)
        conv_blocks = [conv_1, conv_2, conv_3]
        encoder_output = Concatenate()(conv_blocks)

        dense_1 = Dense(1024, activation='relu')(encoder_output)
        drop_1 = Dropout(0.3)(dense_1)
        dense_2 = Dense(1024, activation='relu')(drop_1)
        drop_2 = Dropout(0.3)(dense_2)
        model = Model(input=coco_image, output=drop_2)
        return model

    def language_model(self, seq_length,
                       dict_length, prev_words):
        conv_feat = Input(shape=(1024,))

        conv_repeat = RepeatVector(seq_length)(conv_feat)

        emb = Embedding(dict_length, 300)(prev_words)

        lstm_in = Concatenate()([conv_repeat, emb])

        lstm = LSTM(128, return_sequences=True)(lstm_in)

        predictions = TimeDistributed(Dense(dict_length + 1,
                                            activation='softmax'),
                                      name='out')(lstm)

        model = Model(input=[conv_feat, prev_words],
                      output=predictions)

        return model

    def build_model(self, dict_length):
        seqlen = 20
        coco_image = Input(shape=(128, 128, 3))

        img_emb = self.build_image_model(coco_image)(coco_image)

        prev_words = Input(shape=(seqlen,), name='prev_words')

        lang_model = self.language_model(seqlen, dict_length, prev_words)

        out = lang_model([img_emb, prev_words])

        model = Model(input=[coco_image, prev_words], output=out)

        return model





