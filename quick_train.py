from quick_model import QuickImageCaptioningModel
from skimage import io, transform
from data_generator.data_generator import CocoDataGenerator
from keras.preprocessing.text import Tokenizer
import json
import numpy as np


json_file = '/data/dl_lecture_data/TrainVal/annotations/captions_train2014.json'


def load_train_data():
    json_captions = {}
    with open(json_file, 'r') as f:
        json_captions = json.load(f)
    images = {}
    data_set_images = []
    data_set_annotation = []
    data_set_output = []
    relevant_captions = []
    counter = 0
    for i in json_captions['images']:
        temp_image = io.imread('/data/dl_lecture_data/TrainVal/train2014/' + i['file_name'])
        temp_image = np.array(transform.resize(temp_image, (128, 128)))
        if temp_image.shape != (128, 128, 3):
            continue
        images[int(i['id'])] = temp_image
        counter += 1
        if counter > 100:
            break

    for i in json_captions['annotations']:
        if int(i['image_id']) in images:
            relevant_captions.append(i['caption'])
            data_set_images.append(images[int(i['image_id'])])

    print(data_set_images[0].shape)
    print(len(relevant_captions))

    tokenizer = Tokenizer(num_words=1024)

    tokenizer.fit_on_texts(relevant_captions)

    text_sequences = tokenizer.texts_to_sequences(relevant_captions)

    for text_sequence in text_sequences:
        if len(text_sequence) > 20:
            text_sequence = text_sequence[:20]
        while len(text_sequence) < 20:
            text_sequence.append(0)
        data_set_annotation.append(text_sequence)
        temp_output = text_sequence[1:]
        temp_output.append(0)
        one_hot_output = []
        for ouput in temp_output:
            temp_hot = [0 for i in range(1025)]
            temp_hot[ouput] = 1
            one_hot_output.append(temp_hot)
        data_set_output.append(one_hot_output)

    return data_set_images, data_set_annotation, data_set_output


def main():
    data_gen = CocoDataGenerator(image_limit=20000, batches_per_epoch=50,
                                 images_in_memory=500,
                                 batches_with_images=500)
    data_gen.load_annotation_data()
    data_gen.prepare_captions_for_training()

    model = QuickImageCaptioningModel().build_model(2048)

    model.compile('adam', loss='categorical_crossentropy',
                  sample_weight_mode='temporals')

    model.fit_generator(generator=data_gen, epochs=10,
                        use_multiprocessing=False,
                        workers=8)


if __name__ == "__main__":
    main()
