import numpy as np
import random
import glob
import os.path
from processor import process_image
from keras.utils import to_categorical

class DataSet():
    def __init__(self, seq_length=16, image_shape=(224, 224, 3), dataset_path='/', batch_size=32):
        """Constructor.
        seq_length = (int) the number of frames to consider
        """
        self.seq_length = seq_length
        self.image_shape = image_shape
        self.dataset_path = dataset_path
        self.batch_size = batch_size
                
        # Get the classes.
        self.classes = self.get_classes()

    def get_classes(self):
        """Extract the classes from our data."""
        train_dataset_path = os.path.join(self.dataset_path, 'train')
        classes = os.listdir(train_dataset_path)
        classes = sorted(classes)

        return classes

    def get_class_one_hot(self, class_str):
        """Given a class as a string, return its number in the classes
        list. This lets us encode and one-hot it for training."""
        # Encode it first.
        label_encoded = self.classes.index(class_str)

        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.classes))

        assert len(label_hot) == len(self.classes)

        return label_hot

    def frame_generator(self, train_test):
        """Return a generator that we can use to train on."""
        
        if train_test == 'train':
            self.dir_path = os.path.join(self.dataset_path, 'train')
        else:
            self.dir_path = os.path.join(self.dataset_path, 'test')

        while 1:
            X, y = [], []

            # Generate batch_size samples.
            for _ in range(self.batch_size):
                # Reset to be safe.
                sequence = None

                # Get a random sample.
                sample_class = random.choice(self.classes)
                random_choice_class_path = os.path.join(self.dir_path, sample_class)
                
                while True:
                    list_video = os.listdir(random_choice_class_path)
                    sample_video = random.choice(list_video)
                    sample_video_path = os.path.join(random_choice_class_path, sample_video)
                    
                    images = sorted(glob.glob(os.path.join(sample_video_path, '*jpg')))
                    
                    if(len(images)) >= self.seq_length:
                        break
                        
                frames = self.rescale_list(images, self.seq_length)
                
                # Build the image sequence
                sequence = self.build_image_sequence(frames)

                X.append(sequence)
                y.append(self.get_class_one_hot(sample_class))

            yield np.array(X), np.array(y)

    def build_image_sequence(self, frames):
        """Given a set of frames (filenames), build our sequence."""
        return [process_image(x, self.image_shape) for x in frames]

    def rescale_list(self, input_list, size):
        """Given a list and a size, return a rescaled/samples list. For example,
        if we want a list of size 5 and we have a list of size 25, return a new
        list of size five which is every 5th element of the origina list."""
        assert len(input_list) >= size

        # Get the number to skip between iterations.
        skip = len(input_list) // size

        # Build our new output.
        output = [input_list[i] for i in range(0, len(input_list), skip)]

        # Cut off the last one if needed.
        return output[:size]