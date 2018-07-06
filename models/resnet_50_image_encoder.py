from keras.applications.resnet50 import ResNet50
from keras.layers import AveragePooling2D, Dense


def resnet50_model(img_input, weights=None):
    model = ResNet50(include_top=False, weights='imagenet',
                     input_tensor=img_input)
    x = model.output
    x = AveragePooling2D()(x)
    x = Dense(1024)

    for layer in model.layers:
        layer.trainable = False

    return x, (1024,)
