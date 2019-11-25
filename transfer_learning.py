import os
import pickle
import random
import shutil

import h5py
import numpy as np
from tqdm import tqdm
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)
from keras.preprocessing import image
from keras.applications import imagenet_utils, VGG16, VGG19, DenseNet121
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import classification_report

import file_tools
import image_tools
import hdf5_tools
from networks import SimpleFF


def extract_features(model_name, split_data, output_dir, save_metrics):
    batch_size = 16
    keras_augmentation = True
    my_augmentation = False
    min_train_size = 500
    shuffle_train = True
    padding = 'ruling_gray'
    buffer_size = 1000
    target_shape = (128, 128, 3)
    if keras_augmentation:
        keras_aug = image.ImageDataGenerator(
            shear_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=30,
            zoom_range=0.1,
            width_shift_range=0.05,
            height_shift_range=0.05,
            fill_mode="nearest")
    else:
        keras_aug = None
    if model_name == 'VGG16':
        model = VGG16(include_top=False, weights='imagenet', input_shape=target_shape)
    elif model_name == 'VGG19':
        model = VGG19(include_top=False, weights='imagenet', input_shape=target_shape)
    elif model_name == 'DenseNet':
        model = DenseNet121(include_top=False, weights='imagenet', input_shape=target_shape)
    else:
        raise ValueError(f"Unkown model name: {model_name}")
    features_shape = np.prod(model.layers[-1].output_shape[1:])
    print('feature shape:', features_shape)
    train_X, train_y, test_X, test_y, val_X, val_y, class_names = split_data
    classes = len(class_names)
    if min_train_size > 0:
        train_X, train_y = file_tools.broadcast_samples(train_X, train_y, 
                                                        min_train_size)
    # Prepare image generators
    train_gen = image_tools.img_generator(
        train_X, train_y, classes, batch_size, 
        target_shape, padding, shuffle_train, my_augmentation, keras_aug)
    val_gen = image_tools.img_generator(
        val_X, val_y, classes, batch_size, target_shape, padding)
    test_gen = image_tools.img_generator(
        test_X, test_y, classes, batch_size, target_shape, padding)

    # for i in range(10):
    #     n = next(train_gen)[0][0]
    #     image_tools.plot_img(n)

    db_path = os.path.join(output_dir, 'features.h5')
    print('[INFO] Processing training data.')
    train_writer = hdf5_tools.HDF5Writer(db_path, 
        (len(train_y), features_shape), group_name='train', 
        data_name='features', buf_size=buffer_size)
    extract_in_batches(train_writer, train_gen, np.ceil(len(train_y)/batch_size),
                       model, target_shape, features_shape)
    print('[INFO] Processing validation data.')
    val_writer = hdf5_tools.HDF5Writer(db_path, 
        (len(val_y), features_shape), group_name='validation', 
        data_name='features', buf_size=buffer_size)
    extract_in_batches(val_writer, val_gen, np.ceil(len(val_y)/batch_size),
                       model, target_shape, features_shape)
    print('[INFO] Processing testing data.')
    test_writer = hdf5_tools.HDF5Writer(db_path,
        (len(test_y), features_shape), group_name='test',
        data_name='features', buf_size=buffer_size)
    test_writer.store_class_labels(class_names)
    extract_in_batches(test_writer, test_gen, np.ceil(len(test_y)/batch_size),
                     model, target_shape, features_shape)
    
    if save_metrics:
        with open(os.path.join(output_dir, 'parameters.txt'), 'w') as fh:
            fh.write(f'model: {model_name}\n')
            fh.write(f'min train size: {min_train_size}\n')
            fh.write(f'my augmentation: {my_augmentation}\n')
            fh.write(f'keras augmentation: {keras_augmentation}\n')
            fh.write(f'target shape: {target_shape}\n')
            fh.write(f'feature shape: {features_shape}\n')
            fh.write(f'padding: {padding}\n')
        print(f"[INFO] Features extracted to '{db_path}''")

    return db_path


def extract_in_batches(dataset_writer, data_gen, steps_per_epoch,
                       model, target_shape, features_flattened):
    for _ in tqdm(np.arange(steps_per_epoch)):
        X, y = next(data_gen)
        features = model.predict(X, batch_size=X.shape[0])
        bs, d1, d2, d3 = features.shape
        features = features.reshape((bs, d1*d2*d3))
        y = np.argmax(y, axis=1)
        dataset_writer.add(features, y)
    dataset_writer.close()


def simple_feed_forward(model_name, db_path, output_dir, save_metrics):
    batch_size = 128
    max_epochs = 100
    train_gen = hdf5_tools.hdf5_generator(db_path, 'train', batch_size)
    test_gen = hdf5_tools.hdf5_generator(db_path, 'test', batch_size)
    val_gen = hdf5_tools.hdf5_generator(db_path, 'validation', batch_size)
    with h5py.File(db_path, 'r') as db:
        train_size = len(db['train']['labels'])
        test_size = len(db['test']['labels'])
        val_size = len(db['validation']['labels'])
        classes = len(db['label_names'])
        features_shape = db['test']['features'][0].shape
        if model_name == 'SFF':
            model = SimpleFF().build(features_shape, classes)
        else:
            raise ValueError(f'Unknown model name: {model_name}')
        model.compile(loss='categorical_crossentropy', optimizer='adam', 
                      metrics=['accuracy'])
        print('[INFO] Training Simple Neural Network')
         # Stop training when validation loss hasn't decreased for 6 epochs
        es = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
        # Save only the best model based on validation accuracy
        model_path = os.path.join(output_dir, 'best_model.h5')
        mc = ModelCheckpoint(model_path, monitor='val_acc', mode='max', 
                             save_best_only=True, verbose=1)
        history = model.fit_generator(
            train_gen, 
            steps_per_epoch=np.ceil(train_size/batch_size),
            validation_data=val_gen, 
            validation_steps=np.ceil(val_size/batch_size),
            epochs=max_epochs,
            callbacks=[es, mc])
        print('[INFO] Evaluating network')
        model = load_model(model_path)
        predictions = model.predict_generator(test_gen, 
                                              steps=np.ceil(test_size/batch_size))
        predictions = np.argmax(predictions, axis=1)
        test_report = classification_report(db['test']['labels'], predictions,
                                            target_names=db['label_names'])
    print(test_report)
    if save_metrics:
        with open(os.path.join(output_dir, 'test_report.txt'), 'w') as fh:
            fh.write(test_report)
        image_tools.plot_history(history, os.path.join(output_dir, 
                                 'training_progress.png'))
        print(f"[INFO] Model and metrics saved to '{output_dir}''")
