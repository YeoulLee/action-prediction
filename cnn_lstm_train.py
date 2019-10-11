from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from cnn_lstm_train_data import DataSet
import cnn_lstm_train_model
import time
import os.path
from os import makedirs
import os
from keras.optimizers import SGD, Adam

# train(name_str=name_str, model_path=model_path, seq_length=seq_length, image_shape=image_shape, input_shape=input_shape, batch_size=batch_size, nb_epoch=nb_epoch, dataset_path=dataset_path, out_path=out_path, n_dense=n_dense, p_dropout=p_dropout, freeze=freeze)
def train(name_str = None, model_path='/', seq_length=16, image_shape=(224, 224, 3), input_shape=(None,224,224,3), batch_size=32, nb_epoch=100, dataset_path='/', out_path='/', n_dense=1024, p_dropout=0.5, freeze=True):

    # Get local time.
    time_str = time.strftime("%y%m%d%H%M", time.localtime())

    if name_str == None:
        name_str = time_str

    # Callbacks: Save the model.
    directory1 = os.path.join(out_path, 'checkpoints', name_str)
    if not os.path.exists(directory1):
        os.makedirs(directory1)
        checkpointer = ModelCheckpoint(
            filepath=os.path.join(directory1, 'save_best_only.hdf5'),
            verbose=1,
            save_best_only=True)

    # Callbacks: TensorBoard
    directory2 = os.path.join(out_path, 'TB', name_str)
    if not os.path.exists(directory2):
        os.makedirs(directory2)
    tb = TensorBoard(log_dir=os.path.join(directory2))

    # Callbacks: Early stoper
    early_stopper = EarlyStopping(monitor='loss', patience=100)

    # Callbacks: Save results.
    directory3 = os.path.join(out_path, 'logs', name_str)
    if not os.path.exists(directory3):
        os.makedirs(directory3)
    timestamp = time.time()
    csv_logger = CSVLogger(os.path.join(directory3, 'training-' + \
        str(timestamp) + '.log'))

    
    
    data = DataSet(seq_length,image_shape,dataset_path,batch_size)
    
    # Get generators.
    generator = data.frame_generator('train')
    val_generator = data.frame_generator('test')
    
    model_lrcn = cnn_lstm_train_model.get_model(data,seq_length,input_shape,n_dense,p_dropout,model_path,freeze)
    # Now compile the network.
    #optimizer = Adam(lr=1e-5, decay=1e-6)
    model_lrcn.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy', 'top_k_categorical_accuracy'])
    
    model_lrcn.fit_generator(
        generator,
        steps_per_epoch=100,
        validation_data=val_generator,
        validation_steps=10,
        epochs=nb_epoch,
        callbacks=[tb, early_stopper, csv_logger, checkpointer])
            
        
        
def main():
    """These are the main training settings. Set each before running
    this file."""
    "=============================================================================="
    name_str = 'lcrn_adam'
    model_path='/home/iyeoul/actionprediction/out/rgb_cnn/checkpoints/1910090834/save_best_only.hdf5'
    seq_length=16
    image_shape=(224, 224, 3)
    input_shape=(None,224,224,3)
    batch_size = 32
    nb_epoch = 120000
    dataset_path = '/home/iyeoul/UCF101/rgb'
    out_path='/home/iyeoul/actionprediction/out/cnn_lstm'
    n_dense=1024
    p_dropout=0.5
    freeze=True
    "=============================================================================="

    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    train(name_str=name_str, model_path=model_path, seq_length=seq_length, image_shape=image_shape, input_shape=input_shape, batch_size=batch_size, nb_epoch=nb_epoch, dataset_path=dataset_path, out_path=out_path, n_dense=n_dense, p_dropout=p_dropout, freeze=freeze)

if __name__ == '__main__':
    main()
