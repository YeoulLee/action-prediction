import os
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

class DataSet():
    def __init__(self, dataset_path='/', image_shape=(224, 224), batch_size=32):
        
        self.dataset_path = dataset_path
        self.image_shape = image_shape
        self.batch_size = batch_size
        
        self.classes = self.get_classes()
        
    def get_classes(self):
        train_dataset_path = os.path.join(self.dataset_path, 'train')
        classes = os.listdir(train_dataset_path)
        classes = sorted(classes)
        
        return classes
    
    def get_generators(self):
        train_datagen = ImageDataGenerator(
                rescale=1./255,
                shear_range=0.2,
                horizontal_flip=True,
                rotation_range=10.,
                width_shift_range=0.2,
                height_shift_range=0.2)

        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
                os.path.join(self.dataset_path, 'train'),
                target_size=self.image_shape,
                batch_size=self.batch_size,
                classes=self.classes,
                class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
                os.path.join(self.dataset_path, 'test'),
                target_size=self.image_shape,
                batch_size=self.batch_size,
                classes=self.classes,
                class_mode='categorical')

        return train_generator, validation_generator