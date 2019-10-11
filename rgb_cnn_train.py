from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, CSVLogger
from rgb_cnn_data import DataSet
import rgb_cnn_train_inceptionV3
import rgb_cnn_train_resnet50
import time
import os.path
from os import makedirs
import os

def train_model(model, nb_epoch, generators, callbacks=[]):
    train_generator, validation_generator = generators
    model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        validation_data=validation_generator,
        validation_steps=10,
        epochs=nb_epoch,
        callbacks=callbacks)
    return model

def train(saved_weights=None, image_shape=(224, 224), batch_size=32, nb_epoch=100, name_str=None, dataset_path='/', out_path='/', n_dense=1024, p_dropout=0.5, model='inceptionV3'):

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

    
    
    data = DataSet(dataset_path,image_shape,batch_size)
    
    # Get generators.
    generators = data.get_generators()
    
    

    if model == 'inceptionV3':
        
        # Get the model.
        model_inceptionv3 = rgb_cnn_train_inceptionV3.get_model(data)

        if saved_weights is None:
            print("Loading network from ImageNet weights.")
            print("Get and train the top layers...")
            model_inceptionv3 = rgb_cnn_train_inceptionV3.freeze_all_but_top(model_inceptionv3)
            model = train_model(model_inceptionv3, 10, generators)
        else:
            print("TO DO")

        print("Get and train the mid layers...")
        model_inceptionv3 = rgb_cnn_train_inceptionV3.freeze_all_but_mid_and_top(model_inceptionv3)
        model_inceptionv3 = train_model(model_inceptionv3, nb_epoch, generators, [tb, early_stopper, csv_logger, checkpointer])
        
        
    if model == 'resnet50':
        
        # Get the model.
        model_resnet50 = rgb_cnn_train_resnet50.get_model(data,n_dense,p_dropout)
        
        if saved_weights is None:
            model_resnet50 = train_model(model_resnet50, nb_epoch, generators, [tb, early_stopper, csv_logger, checkpointer])
        else:
            print("TO DO")
            
        
        
def main():
    """These are the main training settings. Set each before running
    this file."""
    "=============================================================================="
    name_str = None
    saved_weights = None
    image_shape=(224, 224)
    batch_size = 64
    nb_epoch = 200
    dataset_path = '/home/iyeoul/UCF101/rgb'
    out_path='/home/iyeoul/actionprediction/out/rgb_cnn'
    n_dense=1024
    p_dropout=0.5
    model='inceptionV3'
    "=============================================================================="

    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    train(saved_weights=saved_weights, image_shape=image_shape, batch_size=batch_size,
            nb_epoch=nb_epoch, name_str=name_str, dataset_path=dataset_path, out_path=out_path, n_dense=n_dense, p_dropout=p_dropout, model=model)
    # def train(saved_weights=None, image_shape=(224, 224), batch_size=32, nb_epoch=100, name_str=None, dataset_path='/', out_path='/', n_dense=1024, p_dropout=0.5, model='inceptionV3'):

if __name__ == '__main__':
    main()
