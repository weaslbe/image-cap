from quick_model import QuickImageCaptioningModel

def load_train_data():
    pass

def main():
    short_train_data = load_train_data()

    model = QuickImageCaptioningModel().build_model(1024)

    model.compile('adam', loss='categorical_crossentropy', sample_weight_mode='temporal')


if __name__ == "__main__":
    main()
