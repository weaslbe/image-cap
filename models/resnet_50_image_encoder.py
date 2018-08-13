from keras.applications.resnet50 import ResNet50
from keras.layers import AveragePooling2D, Dense, Flatten


def resnet50_model(img_input, voc_size, weights=None):
    model = ResNet50(include_top=False, weights='imagenet',
                     input_tensor=img_input)
    x = model.output
    x = AveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    xt = Dense(voc_size, activation='softmax')(x)

    for layer in model.layers:
        layer.trainable = False

    return x, (512,), xt
