import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class GetData:
    def __init__(self):
        self.batch_size = 128
        self.epochs = 10
        self.IMG_HEIGHT = 400
        self.IMG_WIDTH = 400
        train_dir = os.path.join('data/train')
        validation_dir = os.path.join('data/test')

        # Get folders in train
        folders = ([name for name in os.listdir(train_dir)
                    if os.path.isdir(os.path.join(train_dir, name))])  # get all directories

        # get train total image count from each subfolder
        self.total_train = 0
        for folder in folders:
            self.total_train += len(os.listdir(os.path.join(train_dir, folder)))

        # Get folders in test
        folders = ([name for name in os.listdir(validation_dir)
                    if os.path.isdir(os.path.join(validation_dir, name))])  # get all directories

        # get train total image count from each subfolder
        self.total_val = 0
        for folder in folders:
            self.total_val += len(os.listdir(os.path.join(validation_dir, folder)))

        print("Total training images:", self.total_train)
        print("Total validation images:", self.total_val)

        train_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our training data
        validation_image_generator = ImageDataGenerator(rescale=1. / 255)  # Generator for our validation data

        self.train_data_gen = train_image_generator.flow_from_directory(batch_size=self.batch_size,
                                                                        directory=train_dir,
                                                                        shuffle=True,
                                                                        target_size=(self.IMG_HEIGHT, self.IMG_WIDTH),
                                                                        class_mode='binary')

        self.val_data_gen = validation_image_generator.flow_from_directory(batch_size=self.batch_size,
                                                                           directory=validation_dir,
                                                                           target_size=(
                                                                               self.IMG_HEIGHT, self.IMG_WIDTH),
                                                                           class_mode='binary')
