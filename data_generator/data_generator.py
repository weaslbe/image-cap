from keras import utils
from keras.preprocessing.text import text_to_word_sequence
import json

DEFAULT_DIR_PATH = '/data/dl_lecture_data/TrainVal/'


class CocoDataGenerator(utils.Sequence):

    def __init__(self, batch_size=16, images_in_memory=100,
                 batches_with_images=10, directory_path=None,
                 dictionary_size=1024, sequence_length=20,
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
            caption_sequence = text_to_word_sequence(annotation['caption'])
            self.caption_mapping[annotation_id] = caption_sequence

        print(len(self.image_mappings.keys()))
        print(len(self.caption_mapping.keys()))

        max_len = 0
        min_len = 1000
        sum_len = 0
        for key, value in self.caption_mapping.items():
            max_len = max(max_len, len(value))
            min_len = min(min_len, len(value))
            sum_len += len(value)

        print(max_len)
        print(min_len)
        print(sum_len)

    def fetch_new_images(self):
        pass

    def generate_batch(self):
        pass
