import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Average, GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam
from keras.layers.normalization import BatchNormalization

# CNN model for the spatial stream
def get_model(data, n_dense=1024, p_dropout=0.5, weights='imagenet'):
    # create the base pre-trained model
    base_model = ResNet50(weights=weights, include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(n_dense, activation='relu')(x)
    
    # add a dropout
    x = Dropout(p_dropout)(x)
    
    # and a logistic layer
    predictions = Dense(len(data.classes), activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.summary()
    
    model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy', 'top_k_categorical_accuracy'])
    
    return model