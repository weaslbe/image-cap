from data_generator.data_generator import CocoDataGenerator


if __name__ == "__main__":
    gen = CocoDataGenerator()
    gen.load_annotation_data()
    gen.prepare_captions_for_training()
