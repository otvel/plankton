import os
import random
import shutil
from itertools import groupby

import h5py
from sklearn.preprocessing import LabelEncoder


def combined_shuffle(list1, list2):
    combined = list(zip(list1, list2))
    random.shuffle(combined)
    return zip(*combined)


def list_files(directory, filetype=None, extensions=(), min_N=None, max_N=None):
    """Accepted filetypes: image"""

    filetype_extensions = {'image': 
                           ('jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff')}
    if not filetype and not extensions:
        raise ValueError('Must provide a file type or list of extensions')
    if filetype:
        try:
            extensions = filetype_extensions[filetype]
        except KeyError:
            raise ValueError(f'Unknown filetype: {filetype}')
    for dirpath, dirnames, filenames in os.walk(directory):
        if min_N and len(filenames) < min_N:
            continue
        if max_N and len(filenames) > max_N:
            continue
        for filename in filenames:
            file_extension = filename.split('.')[-1].lower()
            if file_extension in extensions:
                file_path = os.path.join(dirpath, filename)
                yield file_path


def parse_labels(dataset, min_N=None, max_N=None, exlude=[]):
    labels = []
    start_depth = dataset.count(os.path.sep)
    for dirpath, dirnames, filenames in os.walk(dataset):
        dirname = os.path.basename(dirpath)
        # Ignore folders below dataset root 
        current_depth = dirpath.count(os.path.sep) - start_depth
        if dirname in exlude or current_depth > 1:
            continue
        if not min_N and not max_N:
            labels.append(dirname)
        elif not max_N:
            if len(filenames) >= min_N:
                labels.append(dirname)
        elif not min_N:
            if len(filenames) <= max_N:
                labels.append(dirname)
        else:
            if len(filenames) >= min_N and len(filenames) <= max_N:
                labels.append(dirname)
    return labels


def copy_dataset(original_dataset, new_dataset, labels):
    for dirpath, dirnames, filenames in os.walk(original_dataset):
        label = os.path.basename(dirpath)
        if label in labels and filenames:
            label = label.replace(' ', '_')
            dst_dir = os.path.join(new_dataset, label)
            os.makedirs(dst_dir)
            print(f'\tCopying files from {dirpath}')
            for i, filename in enumerate(filenames, start=1):
                src = os.path.join(dirpath, filename)
                file_ext = filename.split('.')[-1]
                dst = os.path.join(dst_dir, f'{label}_{i}.{file_ext}')
                shutil.copyfile(src, dst)


def train_test_split(dataset_path, split, min_N, max_N, info_file,
                     split_to_dir=False, split_dir=None, override=False):
    """Split dataset into lists of paths or seperate folders"""

    train_split, test_split, val_split = split
    train_paths = []
    test_paths = []
    val_paths = []
    if split_to_dir:
        if os.path.isdir(split_dir):
            if override:
                shutil.rmtree(split_dir)
            else:
                raise FileExistsError(split_dir, 
                    'exists! Set override to True in order to bypass this.')
        os.makedirs(split_dir)
        train_dir = os.path.join(split_dir, 'train')
        test_dir = os.path.join(split_dir, 'test')
        val_dir = os.path.join(split_dir, 'validation')
        os.makedirs(train_dir)
        os.makedirs(test_dir)
        os.makedirs(val_dir)
    fh = open(info_file, 'w')
    for label in os.listdir(dataset_path):
        label_paths = list(list_files(os.path.join(dataset_path, label), 
                                      'image', min_N=min_N, max_N=max_N))
        if not label_paths:
            continue
        train_stop = int(len(label_paths) * train_split)
        test_stop = train_stop + int(len(label_paths) * test_split)
        random.shuffle(label_paths)
        label_train = label_paths[:train_stop]
        label_test = label_paths[train_stop:test_stop]
        train_paths.extend(label_train)
        test_paths.extend(label_test)
        if val_split:
            label_val = label_paths[test_stop:]
            val_paths.extend(label_val)
        if split_to_dir:
            label_train_dir = os.path.join(train_dir, label)
            os.makedirs(label_train_dir)
            for path in label_train:
                img_name = os.path.basename(path)
                shutil.copyfile(path, os.path.join(label_train_dir, img_name))
            label_test_dir = os.path.join(test_dir, label)
            os.makedirs(label_test_dir)
            for path in label_test:
                img_name = os.path.basename(path)
                shutil.copyfile(path, os.path.join(label_test_dir, img_name))
            if val_split:
                label_val_dir = os.path.join(val_dir, label)
                os.makedirs(label_val_dir)
                for path in label_val:
                    img_name = os.path.basename(path)
                    shutil.copyfile(path, os.path.join(label_val_dir, img_name))
        print(f'{label}, total: {len(label_paths)}, '
              f'train: {len(label_train)} ({len(label_train)/len(label_paths):.3f}), '
              f'test: {len(label_test)} ({len(label_test)/len(label_paths):.4f}) ', 
              f'val: {len(label_val)} ({len(label_val)/len(label_paths):.4f})', 
              file=fh)
    fh.close()
    if split_to_dir:
        return train_dir, test_dir, val_dir
    else:
        random.shuffle(train_paths)
        random.shuffle(test_paths)
        random.shuffle(val_paths)
        return train_paths, test_paths, val_paths


def broadcast_samples(data, labels, goal):
    new_data = []
    new_labels = []
    # Join data and labels as tuples
    combined = zip(data, labels)
    # Group tuples by class
    combined = sorted(combined, key=lambda x: x[1])
    classes = [list(x) for _, x in groupby(combined, lambda x: x[1])]
    # Count size of each class (c) and broadcast data if below goal
    for c in classes:
        i = 0
        while len(c) < goal:
            c.append(c[i])
            i += 1
        class_data, class_labels = zip(*c)
        new_data.extend(class_data)
        new_labels.extend(class_labels)
    new_data, new_labels = combined_shuffle(new_data, new_labels)
    return new_data, new_labels


def old_broadcast_samples(samples, goal):
    i = 0
    while len(samples) < goal:
        samples.append(samples[i])
        i += 1
    return samples


def labels_from_paths(train_paths, test_paths, val_paths=None):
    train_labels = [path.split(os.path.sep)[-2] for path in train_paths]
    test_labels = [path.split(os.path.sep)[-2] for path in test_paths]
    le = LabelEncoder()
    train_labels = le.fit_transform(train_labels)
    test_labels = le.fit_transform(test_labels)
    if val_paths:
        val_labels = [path.split(os.path.sep)[-2] for path in val_paths]
        val_labels = le.fit_transform(val_labels)
        return train_labels, test_labels, val_labels, le.classes_
    return train_labels, test_labels, le.classes_
