from keras.models import Sequential, load_model, Model
from keras.layers import Input, Average, GlobalAveragePooling2D, TimeDistributed, LSTM
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

def get_model(data, seq_length=16, input_shape=(None,224,224,3), n_dense=1024, p_dropout=0.5, model_path='/', freeze=True):
    
    pretrained = load_model(model_path, compile=False)
    
    pretrained.layers.pop()
    pretrained_pop = Model(pretrained.input, pretrained.layers[-1].output)
    pretrained_pop.summary()
    
    if freeze:
        for layer in pretrained_pop.layers[:]:
            layer.trainable = False

    video = Input(shape=input_shape, name='video_input')
    encoded_frame = TimeDistributed(pretrained_pop)(video)
    encoded_vid = LSTM(n_dense, dropout=p_dropout)(encoded_frame)
    outputs = Dense(len(data.classes), activation='softmax')(encoded_vid)
    model = Model(inputs=video, outputs=outputs)
          
    for layer in model.layers:
        print(layer, layer.trainable)        
    
    model.summary()
    
    return model