from data_generator.data_generator import CocoDataGenerator
from models.image_captioning_model import ImageCaptioningModel
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard, ModelCheckpoint


if __name__ == "__main__":
    data_gen = CocoDataGenerator(image_limit=2000, batches_per_epoch=50,
                                 images_in_memory=500,
                                 batches_with_images=500,
                                 image_shape=(224, 224),
                                 dictionary_size=None
                                 )
    data_gen.load_annotation_data()
    data_gen.prepare_captions_for_training()
    data_gen.prebuild_training_files()
    rev_word_index = {}
    for key, value in data_gen.tokenizer.word_index.items():
        rev_word_index[value] = key

    model_wrapper = ImageCaptioningModel(sequence_length=20,
                                         dictionary_length=data_gen.start_token_index,
                                         image_shape=(224, 224),
                                         rev_word_index)

    model = model_wrapper.build_model()

    multi_gpu = multi_gpu_model(model)

    checkpoint_callback = ModelCheckpoint('checkpoint_weights.{epoch:02d}-{val_loss:.2f}.hdf5',
                                          monitor='loss', mode='min', period=1)

    tb_callback = TensorBoard(
        log_dir='./logs', histogram_freq=0, batch_size=16, write_graph=True, write_grads=True,
        write_images=False, embeddings_freq=0, embeddings_layer_names=None,
        embeddings_metadata=None, embeddings_data=None
    )

    multi_gpu.compile('adam', loss='categorical_crossentropy',
                      sample_weight_mode='temporals')

    multi_gpu.fit_generator(generator=data_gen, epochs=10,
                            use_multiprocessing=True,
                            workers=20,
                            callbacks=[tb_callback, checkpoint_callback])

    model.save_weights('new_weights.hf5')

    token_sequence = model_wrapper.generate_caption('', model)
    caption = data_gen.token_sequence_to_sentence(token_sequence)
