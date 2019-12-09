import os
import pickle
import shutil

import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.preprocessing import image
from sklearn.metrics import classification_report

import file_tools
import image_tools
import transfer_learning
from networks import MyLeNet, MyCNN, save_model_structure


def prepare_dataset(old, new, dirs_excluded):
    print('[INFO] Preparing labels')
    labels = file_tools.parse_labels(old, min_N=None, max_N=None, 
                                     exlude=dirs_excluded)
    print(f'\t{len(labels)} classes in total')
    print('[INFO] Preparing dataset')
    os.makedirs(new, exist_ok=True)
    file_tools.copy_dataset(old, new, labels)
    print('[INFO] Dataset is ready!')


def fit_model(model_name, split_data, output_dir, save_metrics):
    # Set values
    max_epochs = 100
    my_augmentation = False
    keras_augmentation = True
    min_train_size = 500
    opt = 'adam'
    shuffle_train = True
    batch_size = 64
    rows = 128
    cols = 128
    chans = 1
    target_shape = (rows, cols, chans)
    padding = 'ruling_gray'
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
    # Extract image paths and labels
    train_X, train_y, test_X, test_y, val_X, val_y, class_names = split_data
    classes = len(class_names)
    # Expand training set if necessary
    if min_train_size > 0:
        train_X, train_y = file_tools.broadcast_samples(train_X, train_y, 
                                                        min_train_size)
    # Prepare image generators
    train_gen = image_tools.img_generator(
        train_X, train_y, classes, batch_size, 
        target_shape, padding, shuffle_train, my_augmentation, keras_aug)
    test_gen = image_tools.img_generator(
        test_X, test_y, classes, batch_size, target_shape, padding)
    val_gen = image_tools.img_generator(
        val_X, val_y, classes, batch_size, target_shape, padding)
    
    # for i in range(2):
    #     n = next(train_gen)[0][0]
    #     image_tools.plot_img(n)

    print('[INFO] Preparing Model.')
    if model_name == 'MyLeNet':
        model = MyLeNet().build(rows, cols, chans, classes)
    elif model_name == 'MyCNN':
        model = MyCNN().build(rows, cols, chans, classes)
    else:
        raise ValueError(f'Unknown model name: {model_name}')

    model.compile(loss='categorical_crossentropy', optimizer=opt,
                  metrics=['accuracy'])
    # Stop training when validation loss hasn't decreased for 15 epochs
    es = EarlyStopping(monitor='val_loss', mode='min', patience=5, verbose=1)
    # Save only the best model based on validation accuracy
    model_path = os.path.join(output_dir, 'best_model.h5')
    mc = ModelCheckpoint(model_path, monitor='val_acc', mode='max', 
                         save_best_only=True, verbose=1)
    print('[INFO] Training Model.')
    history = model.fit_generator(
        train_gen, 
        steps_per_epoch=np.ceil(len(train_y)/batch_size), 
        validation_data=val_gen, 
        validation_steps=np.ceil(len(val_y)/batch_size),
        epochs=max_epochs,
        callbacks=[es, mc])

    print('[INFO] Evaluating network.')
    model = load_model(model_path)
    predictions = model.predict_generator(
        test_gen, 
        steps=np.ceil(len(test_y)/batch_size))
    predictions = np.argmax(predictions, axis=1)
    test_report = classification_report(test_y, predictions, 
                                        target_names=class_names)
    print(test_report)
    if save_metrics:
        with open(os.path.join(output_dir, 'parameters.txt'), 'w') as fh:
            fh.write(f'min_train_size: {min_train_size}\n')
            fh.write(f'my_augmentation: {my_augmentation}\n')
            fh.write(f'keras_augmentation: {keras_augmentation}\n')
            fh.write(f'target_shape: {target_shape}\n')
            fh.write(f'padding: {padding}\n')
            fh.write(f'max_epochs: {max_epochs}\n')
            fh.write(f'batch_size: {batch_size}\n')
            fh.write(f'optimizer: {opt}')
        with open(os.path.join(output_dir, 'test_report.txt'), 'w') as fh:
            fh.write(test_report)
        image_tools.plot_history(history, os.path.join(output_dir, 
                                 'training_progress.png'))
        save_model_structure(model, 
                             os.path.join(output_dir, 'network_summary.txt'))
        print(f"[INFO] Model and metrics saved to '{output_dir}''")


def get_split_data(dataset, split_pickle=None, new=False):
    if new:
        split = (0.55, 0.25, 0.20)
        min_N = 100
        max_N = None
        dist_file = 'output/class_distribution.txt'
        name_file = 'output/class_names.txt'
        train_paths, test_paths, val_paths = file_tools.train_test_split(
            dataset, split, min_N, max_N, dist_file)
        train_labels, test_labels, val_labels, class_names = \
            file_tools.labels_from_paths(train_paths, test_paths, val_paths)
        split_data = (train_paths, train_labels, 
                      test_paths, test_labels,
                      val_paths, val_labels, class_names)
        with open(split_pickle, 'wb') as fh:
            pickle.dump(split_data, fh)
        with open(name_file, 'w') as fh:
            for name in class_names:
                fh.write(f'{name}\n')
    else:
        with open(split_pickle, 'rb') as fh:
            split_data = pickle.load(fh)
    return split_data


def test_paths_to_images(split_data, test_dir_path):
    test_X = split_data[2]
    os.makedirs(test_dir_path, exist_ok=True)
    for i in range(len(test_X)):
        src = test_X[i]
        label = os.path.basename(src)
        dst = os.path.join(test_dir_path, label)
        shutil.copy(src, dst)


def main():
    """Comment out parts that you don't want to run"""

    # Dataset preparation (56843 images in final syke-dataset) 
    original_dataset = '/home/otso/Datasets/SYKE_150819'
    new_dataset = '/home/otso/Datasets/syke'
    dirs_excluded = ['summary', 'Unclassified']
    prepare_dataset(original_dataset, new_dataset, dirs_excluded)

    # Split image paths
    dataset = '/home/otso/Datasets/syke'
    split_data_path = 'output/split_data.pickle'
    split_data = get_split_data(dataset, split_data_path, new=False)
    
    # Copy test images to separate directory for testing purposes
    test_paths_to_images(split_data, '/home/otso/Datasets/syke_test')
    
    # Define model path
    output_dir = 'models/test'
    os.makedirs(output_dir, exist_ok=True)
    save_metrics = True

    # Train MyLeNet or MyCNN
    fit_model('MyLeNet', split_data, output_dir, save_metrics)

    # Extract features (transfer learning)
    model_name = 'VGG19'
    transfer_learning.extract_features(model_name, split_data, 
                                       output_dir, save_metrics)
    # Train simple neural network on features
    db_path = f'models/feat/{model_name}/features.h5'
    transfer_learning.simple_feed_forward('SFF', db_path, output_dir, save_metrics)

    print('[INFO] Finished!')


if __name__ == '__main__':
    main() 