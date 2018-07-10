import json
from models.image_captioning_model import ImageCaptioningModel


def load_reverted_word_index():
    word_index = {}
    with open('rev_word_index.json', 'r') as f:
        word_index = json.load(f)
    return word_index


def load_parameters():
    parameters = {}
    with open('parameters.json', 'r') as f:
        parameters = json.load(f)
    return parameters


def load_image_json():
    image_json = {}
    with open('test/data/input.json', 'r') as f:
        image_json = json.load(f)
    return image_json


def generate_captions(model_wrapper, model,
                      image_json, reverted_word_index):
    captions = []
    for image in image_json['images']:
        image_save = {}
        t_sequence = model_wrapper.generate_captions(image['file_name'],
                                                     model)
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
        words.append(reverted_word_index[i])
    return ' '.join(words)


def save_submission_file(captions):
    with open('test/pred/results.json', 'w+') as f:
        json.dump(captions, f)


def main():
    reverted_word_index = load_reverted_word_index()
    parameters = load_parameters()
    image_json = load_image_json()

    model_wrapper = ImageCaptioningModel(sequence_length=parameters['sequence_length'],
                                         dictionary_length=parameters['dictionary_length'],
                                         image_shape=parameters['image_shape'],
                                         rev_word_index=reverted_word_index,
                                         res50=parameters['res50'])

    model = model_wrapper.build_model()

    model.load_weights('submission_weights.hdf5')

    model.compile('adam', loss='categorical_crossentropy',
                  sample_weight_mode='temporal')

    captions = generate_captions(model_wrapper, model,
                                 image_json, reverted_word_index)

    save_submission_file(captions)


if __name__ == "__main__":
    main()
