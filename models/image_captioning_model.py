from keras.layers import Conv2D, Input, Concatenate, Dense, Dropout, TimeDistributed, RepeatVector, Embedding, Flatten, LSTM
from keras.models import Model
from models.language_model import LanguageModel
from models.resnet_152_image_encoder import resnet152_model
from models.resnet_50_image_encoder import resnet50_model
from skimage import io, transform

import numpy as np


class ImageCaptioningModel:

    def __init__(self, sequence_length=20, dictionary_length=2048,
                 image_shape=(224, 224), rev_word_index=None, is_local=False,
                 res50=False):
        self.sequence_length = sequence_length
        self.dictionary_length = dictionary_length
        self.image_shape = image_shape
        self.language_model = LanguageModel(self.dictionary_length,
                                            self.sequence_length, pre_build_embedding=True,
                                            reverted_word_index=rev_word_index, is_local=is_local)
        self.is_local = is_local
        if res50:
            self.build_image_model = resnet50_model
        else:
            self.build_image_model = resnet152_model

    def build_model(self):
        coco_image = Input(shape=(self.image_shape[0],
                                  self.image_shape[1],
                                  3))
        resnet_weights_path = 'resnet152_weights_tf.h5'
        if self.is_local:
            resnet_weights_path = "data/" + resnet_weights_path

        img_emb, output_shape = self.build_image_model(coco_image,
                                                       resnet_weights_path)

#        img_emb = img_emb(coco_image)

        prev_words = Input(shape=(self.sequence_length,),
                           name='prev_words')

        lang_model = self.language_model.build_language_model(prev_words,
                                                              img_emb,
                                                              output_shape,
                                                              coco_image)

#        out = lang_model([coco_image, prev_words])

#        model = Model(input=[coco_image, prev_words], output=out)

        return lang_model

    def generate_caption(self, image, model):
        image_array = io.imread(image)
        image_array = transform.resize(image_array,
                                       self.image_shape)
        image_array[:, :, 0] -= 103.939
        image_array[:, :, 1] -= 116.779
        image_array[:, :, 2] -= 123.68
        seq_input = [self.dictionary_length + 1] + [0 for i in range(self.sequence_length - 1)]
        output_sentence = []
        for word_index in range(self.sequence_length):
            output = model.predict([np.array(image_array),
                                    np.array(seq_input)])
            output_token = np.argmax(output[0][word_index])
            output_sentence.append(output_token)
            if output_token == 0:
                while len(output_sentence) < self.sequence_length:
                    output_sentence.append(0)
            if word_index < (self.sequence_length - 1):
                seq_input[word_index + 1] = output_token
        return output_sentence


