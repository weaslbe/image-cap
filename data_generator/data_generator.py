import numpy as np
from keras import utils
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from tqdm import tqdm

from skimage import io, transform
import json

DEFAULT_DIR_PATH = '/data/dl_lecture_data/TrainVal/'

# TODO


class CocoDataGenerator(utils.Sequence):

    def __init__(self, batch_size=16, images_in_memory=200,
                 batches_with_images=50, directory_path=None,
                 dictionary_size=2048, sequence_length=20,
                 image_shape=(128, 128), batches_per_epoch=10,
                 image_limit=None, lazy_build_output=True,
                 load_all=False, predict_series=True,
                 pre_build=True,
                 pre_save_directory='/home/cps6/image-cap/data/'):
        self.batch_size = batch_size
        self.images_in_memory = images_in_memory
        self.batches_with_images = batches_with_images
        self.dictionary_size = dictionary_size
        self.sequence_length = sequence_length
        self.batches_per_epoch = batches_per_epoch
        self.current_batch_counter = 0
        self.image_limit = image_limit
        self.image_shape = image_shape
        self.lazy_build_output = lazy_build_output
        self.load_all = load_all
        self.predict_series = predict_series
        self.pre_build = pre_build
        self.pre_save_directory = pre_save_directory

        if directory_path is None:
            self.directory_path = DEFAULT_DIR_PATH
        else:
            self.directory_path = directory_path

    def __len__(self):
        if not self.pre_build:
            return self.batches_per_epoch
        else:
            return self.batch_counts

    def __getitem__(self, idx):
        if not self.pre_build:
            return self.generate_batch()
        else:
            return self.generate_prebuild_batch(idx)

    def load_annotation_data(self):
        relevant_file = self.directory_path + 'annotations/captions_train2014.json'

        json_annotations = {}
        with open(relevant_file, 'r') as f:
            json_annotations = json.load(f)

        self.image_mappings = {}

        image_counter = 0

        for image_data in json_annotations['images']:
            image_id = int(image_data['id'])
            self.image_mappings[image_id] = (image_data['file_name'], [])
            if self.image_limit is not None:
                if image_counter > self.image_limit:
                    break
            image_counter += 1

        self.caption_mapping = {}

        for annotation in json_annotations['annotations']:
            image_id = int(annotation['image_id'])
            annotation_id = int(annotation['id'])
            if image_id not in self.image_mappings:
                continue
            self.image_mappings[image_id][1].append(annotation_id)
            caption = annotation['caption']
            self.caption_mapping[annotation_id] = [caption, image_id]

    def fetch_new_images(self):
        relevant_directory = self.directory_path + 'train2014/'
        avail_images = np.array(list(self.image_mappings.keys()))
        images_to_load = np.random.choice(avail_images,
                                          size=self.images_in_memory)

        relevant_images = {}

        relevant_annotation_ids = []

        for image_id in images_to_load:
            image_filename = relevant_directory + self.image_mappings[image_id][0]
            image = self.load_image(image_filename)

            if image is None:
                continue

            relevant_images[image_id] = image
            relevant_annotation_ids.extend(self.image_mappings[image_id][1])
        self.relevant_images = relevant_images
        self.relevant_annotation_ids = relevant_annotation_ids

    def fetch_all_images(self):
        self.relevant_images = {}
        self.relevant_annotation_ids = []

        for image_id in list(self.image_mappings):
            image_filename = relevant_directory + self.image_mappings[image_id][0]
            image = self.load_image(image_filename)

            if image is None:
                continue

            self.relevant_images[image_id] = image
            self.relevant_annotation_ids.extend(self.image_mappings[image_id][1])

    def load_image(self, filename):
        image_downloaded = io.imread(filename)
        if image_downloaded.ndim != 3:
            return None
        image_downloaded = np.array(transform.resize(image_downloaded,
                                                     self.image_shape))

        image_downloaded[:, :, 0] -= 103.939
        image_downloaded[:, :, 1] -= 116.779
        image_downloaded[:, :, 2] -= 123.68
        return image_downloaded

    def build_auxillary_loss(self, image_id):
        auxillary_loss = np.zeros(self.start_token_index)
        for annotation in self.image_mappings[image_id][1]:
            for word in self.caption_mapping[annotation][0][0]:
                auxillary_loss[word] = 1
        return auxillary_loss

    def prebuild_training_files(self):
        relevant_directory = self.directory_path + 'train2014/'
        current_batch = [[], [], [], []]
        batch_builder_counter = 0
        self.batch_counts = 0
        for caption_id in tqdm(list(self.caption_mapping)):
            image_id = self.caption_mapping[caption_id][1]
            image_filename = relevant_directory + self.image_mappings[image_id][0]
            image = self.load_image(image_filename)

            if image is None:
                continue

            current_batch[0].append(image)
            current_batch[1].append(self.caption_mapping[caption_id][0][0])
            current_batch[2].append(self.caption_mapping[caption_id][0][1])
            current_batch[3].append(self.build_sample_weights(self.caption_mapping[caption_id][0][0]))
            current_batch[4].append(self.build_auxillary_loss(image_id))

            batch_builder_counter += 1

            if batch_builder_counter >= 16:
                self.save_batch_to_disk(self.batch_counts, current_batch)
                batch_builder_counter = 0
                current_batch = [[], [], [], []]

    def save_batch_to_disk(self, batch_id, batch):
        base_filename = self.pre_save_directory + str(batch_id)
        np.save(base_filename + "_img.npy", np.array(batch[0], dtype='f'))
        np.save(base_filename + "_sen.npy", np.array(batch[1]))
        np.save(base_filename + "_out.npy", np.array(batch[2]))
        np.save(base_filename + "_weights.npy", np.array(batch[3], dtype='f'))
        np.save(base_filename + "_aux_loss.npy", np.array(batch[4]))

    def load_batch_from_disk(self, batch_id):
        base_filename = self.pre_save_directory + str(batch_id)
        img = np.load(base_filename + "_img.npy")
        sen = np.load(base_filename + "_sen.npy")
        out = np.load(base_filename + "_out.npy")
        weights = np.load(base_filename + "_weights.npy")
        return img, sen, out, weights

    def build_sample_weights(self, sentence):
        sen_len = 0
        for word_id in sentence:
            if word_id == 0:
                break
            sen_len += 1
        weight = 1.0 / sen_len
        weights = []
        for i in range(len(sentence)):
            if i >= sen_len:
                weights.append(0)
            else:
                weights.append(weight)
        return weights

    def prepare_captions_for_training(self):
        self.caption_tokenizer = Tokenizer(num_words=self.dictionary_size)

        all_captions = [' '.join(text_to_word_sequence(caption[0])) for key, caption in self.caption_mapping.items()]

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
            if not self.lazy_build_output:
                output = self.build_one_hot_output(caption_output)
            else:
                output = caption_output
            self.caption_mapping[key][0] = (np.array(caption_tokenized),
                                            output)

    def build_one_hot_output(self, sequence):
        output = np.zeros((self.sequence_length, self.start_token_index))
        output[np.arange(self.sequence_length), np.array(sequence)] = 1
        return output

    def generate_batch(self):
        if not self.load_all and self.current_batch_counter == 0:
            self.fetch_new_images()
        captions_for_batch = np.random.choice(self.relevant_annotation_ids,
                                              size=self.batch_size)
        batch_image = []
        batch_caption = []
        batch_output = []
        for caption_id in captions_for_batch:
            image_id = self.caption_mapping[caption_id][1]
            batch_image.append(self.relevant_images[image_id])
            batch_caption.append(self.caption_mapping[caption_id][0][0])
            if not self.lazy_build_output:
                batch_output.append(self.caption_mapping[caption_id][0][1])
            else:
                output = self.build_one_hot_output(self.caption_mapping[caption_id][0][1])
                batch_output.append(output)
        self.current_batch_counter = (self.current_batch_counter + 1) % self.batches_with_images

        return [np.array(batch_image), np.array(batch_caption)], np.array(batch_output)

    def generate_prebuild_batch(self, batch_id):
        batch = self.load_batch_from_disk(batch_id)
        if not self.lazy_build_output:
            output = batch[2]
        else:
            output = []
            for sentence in batch[2]:
                output.append(self.build_one_hot_output(sentence))

        return [batch[0], batch[1]], [np.array(batch[2])], batch[3]

    def token_sequence_to_sentence(self, token_sequence):
        reverted_word_index = {}
        for key, value in self.caption_tokenizer.word_index.items():
            reverted_word_index[value] = key
        words = []
        for i in token_sequence:
            if i == 0:
                break
            words.append(reverted_word_index[i])
        return ' '.join(words)
