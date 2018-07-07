from keras.applications.resnet50 import ResNet50
from keras.layers import AveragePooling2D, Dense, Flatten


def resnet50_model(img_input, weights='imagenet'):
    model = ResNet50(include_top=False, weights=weights,
                     input_tensor=img_input)
    x = model.output
    x = AveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)

    for layer in model.layers:
        layer.trainable = False

    return x, (1024,)
