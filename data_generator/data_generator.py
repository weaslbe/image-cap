from keras.utils import Sequence
import json


class CocoDataGenerator(Sequence):

    def __init__(self, batch_size, images_in_memory,
                 batches_with_images, directory_path,
                 dictionary_size, sequence_length):
        self.batch_size = batch_size
        self.images_in_memory = images_in_memory
        self.batches_with_images = batches_with_images
        self.directory_path = directory_path
        self.dictionary_size = dictionary_size
        self.sequence_length = sequence_length

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
            self.caption_mapping[annotation_id] = annotation['caption']

        print(len(self.image_mappings.keys()))
        print(len(self.caption_mapping.keys()))

    def generate_batch(self):
        pass
