import os

import numpy as np
import h5py

from keras.utils import to_categorical


class HDF5Writer:
    """Class for writing data to HDF5 format"""

    def __init__(self, db_path, dims, group_name=None, data_name='images', 
                 buf_size=1000):
        self.root_db = h5py.File(db_path, 'a')
        if group_name:
            self.db = self.root_db.create_group(group_name)
        else:
            self.db = self.root_db
        self.data = self.db.create_dataset(data_name, dims, dtype='float')
        self.labels = self.db.create_dataset('labels', (dims[0],), dtype='float')
        self.buf_size = buf_size
        self.buffer = {'data': [], 'labels': []}
        self.idx = 0

    def add(self, rows, labels):
        self.buffer['data'].extend(rows)
        self.buffer['labels'].extend(labels)
        if len(self.buffer['data']) >= self.buf_size:
            self.flush()
    
    def flush(self):
        i = self.idx + len(self.buffer['data'])
        self.data[self.idx:i] = self.buffer['data']
        self.labels[self.idx:i] = self.buffer['labels']
        self.idx = i
        self.buffer = {'data': [], 'labels': []}

    def store_class_labels(self, class_labels):
        dt = h5py.special_dtype(vlen=str)
        label_set = self.root_db.create_dataset('label_names',
            (len(class_labels),), dtype=dt)
        label_set[:] = class_labels
    
    def close(self):
        if len(self.buffer['data']) > 0:
            self.flush()
        self.root_db.close()


def hdf5_generator(db_path, group_name, batch_size):
    with h5py.File(db_path, 'r') as db:
        samples = len(db[group_name]['labels'])
        classes = len(db['label_names'])
        i = 0
        while True:
            X = []
            y = []
            for _ in range(batch_size):
                sample_X = db[group_name]['features'][i]
                sample_y = db[group_name]['labels'][i]
                X.append(sample_X)
                y.append(sample_y)
                i += 1
                if i == samples:
                    i = 0
                    break
            X = np.array(X)
            y = to_categorical(y, classes)
            yield (X, y)
