from data_generator.data_generator import CocoDataGenerator
from models.image_captioning_model import ImageCaptioningModel
from keras.utils import multi_gpu_model
from keras.callbacks import TensorBoard, ModelCheckpoint
import json

LOCAL = True

if __name__ == "__main__":
    if LOCAL:
        pre_save_directory = "data/preprocessed/"
        directory_path = "data/"
        local_image_limit = 15

    data_gen = CocoDataGenerator(image_limit=local_image_limit if LOCAL else 120000, batches_per_epoch=50,
                                 images_in_memory=500,
                                 batches_with_images=500,
                                 image_shape=(224, 224),
                                 dictionary_size=10000,
                                 batch_size=8,
                                 sequence_length=12,
                                 directory_path=directory_path,
                                 pre_save_directory=pre_save_directory, is_local=LOCAL
                                 )

    data_gen.load_annotation_data()
    data_gen.prepare_captions_for_training()
    data_gen.prebuild_training_files()
    # if LOCAL && prebuild_training_files() done : comment line above and
    # set data_gen.batch_counts = preprocessed_files count / 5
    #data_gen.batch_counts = 3

    '''val_data_gen = CocoDataGenerator(image_limit=local_image_limit if LOCAL else 120000, batches_per_epoch=50,
                                     images_in_memory=500,
                                     batches_with_images=500,
                                     image_shape=(224, 224),
                                     dictionary_size=None, directory_path=directory_path,
                                     pre_save_directory=pre_save_directory, is_local=LOCAL, is_val=True
                                     )
    val_data_gen.load_annotation_data()
    val_data_gen.prepare_captions_for_training()
    val_data_gen.prebuild_training_files()'''

    rev_word_index = {}
    for key, value in data_gen.caption_tokenizer.word_index.items():
        rev_word_index[value] = key

    with open('data/images_test/rev_word_index.json', 'w+') as f:
        json.dump(rev_word_index, f)

    model_wrapper = ImageCaptioningModel(sequence_length=12,
                                         dictionary_length=data_gen.start_token_index,
                                         image_shape=(224, 224),
                                         rev_word_index=rev_word_index, is_local=LOCAL,
                                         res50=True)

    model = model_wrapper.build_model()
    if not LOCAL:
        model = multi_gpu_model(model, gpus=2)

    checkpoint_callback = ModelCheckpoint('checkpoint_weights.{epoch:02d}.hdf5',
                                          monitor='loss', mode='min', period=1)

    #    tb_callback = TensorBoard(
    #        log_dir='./logs', histogram_freq=1, batch_size=16,
    #        write_graph=True, write_grads=True,
    #        write_images=False, embeddings_freq=0, embeddings_layer_names=None,
    #        embeddings_metadata=None, embeddings_data=None
    #    )

    # model.compile('adam', loss='categorical_crossentropy', sample_weight_mode='temporal')
    model.compile('adam', loss='categorical_crossentropy')

    model.fit_generator(generator=data_gen, epochs=2 if LOCAL else 10,
                        use_multiprocessing=True,
                        workers=20,
                        callbacks=[checkpoint_callback],
                        verbose=2)

    model.save_weights('new_weights.hdf5')

    '''token_sequence = model_wrapper.generate_caption('data/images_test/train2014/COCO_train2014_000000057870.jpg', model)
    print(f"token sequence is: {token_sequence}")
    caption = data_gen.token_sequence_to_sentence(token_sequence)
    print(f"caption is: {caption}")'''
