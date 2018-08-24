import json
import pickle
from models.image_captioning_model import ImageCaptioningModel
from main import LOCAL


def load_reverted_word_index():
    word_index = {}
    if LOCAL:
        rev_word_index_path = 'data/images_test/rev_word_index.json'
    else:
        rev_word_index_path = 'rev_word_index.json'
    with open(rev_word_index_path, 'r') as f:
        word_index = json.load(f)
    return word_index


def load_parameters():
    parameters = {}
    if LOCAL:
        parameters_path = 'data/images_test/parameters.json'
    else:
        parameters_path = 'parameters.json'
    with open(parameters_path, 'r') as f:
        parameters = json.load(f)
    return parameters


def load_image_json():
    image_json = {}
    if LOCAL:
        test_data_path = "data/images_test/test/test_data.json"
    else:
        test_data_path = 'test/data/input.json'
    with open(test_data_path, 'r') as f:
        image_json = json.load(f)
    return image_json


def generate_captions(model_wrapper, model,
                      image_json, reverted_word_index):
    captions = []
    for image in image_json['images']:
        image_save = {}
        try:
            if LOCAL:
                test_data_path = 'data/images_test/test/data/'
            else:
                test_data_path = 'test/data/'
            t_sequence = model_wrapper.generate_caption(test_data_path + image['file_name'],
                                                        model)
        except:
            print(image['file_name'])
            t_sequence = [0 for i in range(20)]
        caption = token_sequence_to_caption(t_sequence, reverted_word_index)
        image_save['image_id'] = image['id']
        image_save['caption'] = caption
        captions.append(image_save)
    return captions


def token_sequence_to_caption(token_sequence, reverted_word_index):
    words = []
    for i in token_sequence:
        if i == 0:
            break
        words.append(reverted_word_index[str(i)])
    return ' '.join(words)


def save_submission_file(captions):
    if LOCAL:
        results_path = 'data/images_test/results.json'
    else:
        results_path = 'test/pred/results.json'
    with open(results_path, 'w+') as f:
        json.dump(captions, f)


def main():
    reverted_word_index = load_reverted_word_index()
    parameters = load_parameters()
    image_json = load_image_json()

    model_wrapper = ImageCaptioningModel(sequence_length=parameters['sequence_length'],
                                         dictionary_length=parameters['dictionary_length'],
                                         image_shape=parameters['image_shape'],
                                         rev_word_index=reverted_word_index, is_local=LOCAL,
                                         res50=parameters['res50'])

    model = model_wrapper.build_model()

    if LOCAL:
        model.load_weights('new_weights.hdf5')
    else:
        model.load_weights('submission_weights.hdf5')

    model.compile('adam', loss='categorical_crossentropy',
                  sample_weight_mode='temporal')

    captions = generate_captions(model_wrapper, model,
                                 image_json, reverted_word_index)

    save_submission_file(captions)


if __name__ == "__main__":
    main()
