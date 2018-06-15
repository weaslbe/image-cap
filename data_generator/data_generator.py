from keras import utils
from keras.preprocessing.text import Tokenizer
from skimage import io, transform

import numpy as np
import json

DEFAULT_DIR_PATH = '/data/dl_lecture_data/TrainVal/'


class CocoDataGenerator(utils.Sequence):

    def __init__(self, batch_size=16, images_in_memory=200,
                 batches_with_images=50, directory_path=None,
                 dictionary_size=2048, sequence_length=20,
                 image_shape=(128, 128), batches_per_epoch=10):
        self.batch_size = batch_size
        self.images_in_memory = images_in_memory
        self.batches_with_images = batches_with_images
        self.dictionary_size = dictionary_size
        self.sequence_length = sequence_length
        self.batches_per_epoch = batches_per_epoch
        self.current_batch_counter = 0

        if directory_path is None:
            self.directory_path = DEFAULT_DIR_PATH
        else:
            self.directory_path = directory_path

    def __len__(self):
        return self.batches_per_epoch

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
            self.caption_mapping[annotation_id] = [caption, image_id]

    def fetch_new_images(self):
        relevant_directory = self.directory_path + 'train2014/'
        images_to_load = np.random.choice(self.image_mappings.keys(),
                                          size=self.images_in_memory)

        relevant_images = {}

        relevant_annotation_ids = []

        for image_id in images_to_load:
            image_filename = relevant_directory + self.image_mappings[image_id][0]
            image_downloaded = io.imread(image_filename)
            if image_downloaded.shape[2] != 3:
                continue
            image_downloaded = np.array(transform.resize(image_downloaded,
                                                         self.image_shape))
            relevant_images[image_id] = image_downloaded
            relevant_annotation_ids.extend(self.image_mappings[image_id][1])
        self.relevant_images = relevant_images
        self.relevant_annotation_ids = relevant_annotation_ids

    def prepare_captions_for_training(self):
        self.caption_tokenizer = Tokenizer(num_words=self.dictionary_size)

        all_captions = [caption[0] for key, caption in self.caption_mapping.items()]

        self.caption_tokenizer.fit_on_texts(all_captions)

        if not self.dictionary_size:
            self.start_token_index = len(self.caption_tokenizer.word_index) + 1
        else:
            self.start_token_index = self.dictionary_size + 1
        self.end_token_index = 0

        for key, caption in list(self.caption_mapping.items()):
            caption = caption[0]
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
                num_tokens = self.start_token_index
                one_hot = [0 for i in range(num_tokens)]
                one_hot[word_token] = 1
                caption_one_hot.append(one_hot)
            self.caption_mapping[key][0] = (np.array(caption_tokenized), np.array(caption_output))

    def generate_batch(self):
        if self.current_batch_counter == 0:
            self.fetch_new_images
        captions_for_batch = np.random.choice(self.relevant_annotation_ids,
                                              size=self.batch_size)
        batch_image = []
        batch_caption = []
        batch_output = []
        for caption_id in captions_for_batch:
            image_id = self.caption_mapping[caption_id][1]
            batch_image.append(self.relevant_images[image_id])
            batch_caption.append(self.caption_mapping[caption_id][0][0])
            batch_output.append(self.caption_mapping[caption_id][0][1])
        self.current_batch_counter = (self.current_batch_counter + 1) % self.batches_with_images

        return [np.array(batch_image), np.array(batch_caption)], np.array(batch_output)
