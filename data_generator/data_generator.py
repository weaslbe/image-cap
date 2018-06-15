from keras import utils
from keras.preprocessing.text import Tokenizer

import numpy as np
import json

DEFAULT_DIR_PATH = '/data/dl_lecture_data/TrainVal/'


class CocoDataGenerator(utils.Sequence):

    def __init__(self, batch_size=16, images_in_memory=100,
                 batches_with_images=10, directory_path=None,
                 dictionary_size=None, sequence_length=20,
                 image_shape=(128, 128)):
        self.batch_size = batch_size
        self.images_in_memory = images_in_memory
        self.batches_with_images = batches_with_images
        self.dictionary_size = dictionary_size
        self.sequence_length = sequence_length

        if directory_path is None:
            self.directory_path = DEFAULT_DIR_PATH
        else:
            self.directory_path = directory_path

    def __len__(self):
        return self.batches_with_images

    def __get_item__(self):
        return self.generate_batch()

    def load_annotation_data(self):
        relevant_file = self.directory_path + 'annotations/captions_train2014.json'

        json_annotations = {}
        with open(relevant_file, 'r') as f:
            json_annotations = json.load(f)

        self.image_mappings = {}

        for image_data in json_annotations['images']:
            image_id = int(image_data['id'])
            self.image_mappings[image_id] = (image_data['file_name'], [])

        self.caption_mapping = {}

        for annotation in json_annotations['annotations']:
            image_id = int(annotation['image_id'])
            annotation_id = int(annotation['id'])
            self.image_mappings[image_id][1].append(annotation_id)
            caption = annotation['caption']
            self.caption_mapping[annotation_id] = caption

    def fetch_new_images(self):
        pass

    def prepare_captions_for_training(self):
        self.caption_tokenizer = Tokenizer(num_words=self.dictionary_size)

        all_captions = [caption for key, caption in self.caption_mapping.items()]

        self.caption_tokenizer.fit_on_texts(all_captions)

        self.start_token_index = len(self.caption_tokenizer.word_index) + 1
        self.end_token_index = 0

        for key, caption in list(self.caption_mapping.items()):
            caption_tokenized = self.caption_tokenizer.texts_to_sequences([caption])[0]
            if len(caption_tokenized) > self.sequence_length - 1:
                caption_tokenized = caption_tokenized[:self.sequence_length - 1]
            caption_tokenized = [self.start_token_index] + caption_tokenized
            while len(caption_tokenized) < self.sequence_length:
                caption_tokenized.append(self.end_token_index)
            caption_output = caption_tokenized[1:]
            caption_output.append(self.end_token_index)
            caption_one_hot = []
            for word_token in caption_output:
                num_tokens = len(self.caption_tokenizer.word_index) + 1
                one_hot = [0 for i in range(num_tokens)]
                one_hot[word_token] = 1
                caption_one_hot.append(one_hot)
            self.caption_mapping[key] = (np.array(caption_tokenized), np.array(caption_output))

    def generate_batch(self):
        pass
