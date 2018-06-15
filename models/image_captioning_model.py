from keras.layers import Conv2D, Input, Concatenate, Dense, Dropout, TimeDistributed, RepeatVector, Embedding, Flatten, LSTM
from keras.models import Model
from models.language_model import LanguageModel
from models.resnet_152_image_encoder import resnet152_model


class ImageCaptioningModel:

    def __init__(self, sequence_length=20, dictionary_length=2048,
                 image_shape=(224, 224)):
        self.sequence_length = sequence_length
        self.dictionary_length = dictionary_length
        self.image_shape = image_shape
        self.language_model = LanguageModel(self.dictionary_length,
                                            self.sequence_length,
                                            ())
        self.build_image_model = resnet_15

    def build_model(self):
        coco_image = Input(shape=(self.image_shape[0],
                                  self.image_shape[1],
                                  3))

        img_emb, output_shape = self.build_image_model(coco_image)(coco_image)

        prev_words = Input(shape=(self.sequence_length,),
                           name='prev_words')

        lang_model = self.language_model.build_language_model(prev_words,
                                                              output_shape)

        out = lang_model([img_emb, prev_words])

        model = Model(input=[coco_image, prev_words], output=out)

        return model
