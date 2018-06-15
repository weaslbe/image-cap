from data_generator.data_generator import CocoDataGenerator
from models.image_captioning_model import ImageCaptioningModel


if __name__ == "__main__":
    data_gen = CocoDataGenerator(image_limit=2000, batches_per_epoch=50,
                                 images_in_memory=500,
                                 batches_with_images=500,
                                 image_shape=(224, 224))
    data_gen.load_annotation_data()
    data_gen.prepare_captions_for_training()

    model_wrapper = ImageCaptioningModel(sequence_length=20,
                                         dictionary_length=1024,
                                         image_shape=(224, 224))

    model = model_wrapper.build_model()

    model.compile('adam', loss='categorical_crossentropy',
                  sample_weight_mode='temporals')

    model.fit_generator(generator=data_gen, epochs=10,
                        use_multiprocessing=False,
                        workers=1)

    token_sequence = model_wrapper.generate_caption('', model)
    caption = data_gen.token_sequence_to_sentence(token_sequence)
    print(caption)
