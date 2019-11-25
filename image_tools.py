import os
import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras.utils import to_categorical

import file_tools


def img_generator(img_paths, img_labels, n_classes, 
                  batch_size, target_shape, padding=None, 
                  shuffle=False, my_aug=False, keras_aug=None):
    """Generates batches of images and labels"""

    if len(target_shape) != 3:
        raise ValueError(f'Non valid target shape: {target_shape}')
    nb_samples = len(img_labels)
    i = 0
    while True:
        X = []
        y = []
        for _ in range(batch_size):
            img = load_img(img_paths[i], target_shape, padding, my_aug)
            img = np.reshape(img, target_shape)
            img = img.astype('float32') / 255
            label = img_labels[i]
            X.append(img)
            y.append(label)
            i += 1
            if i == nb_samples:
                if shuffle:
                    img_paths, img_labels = file_tools.combined_shuffle(
                        img_paths, img_labels)
                i = 0
                break
        X = np.array(X)
        y = to_categorical(y, n_classes)
        if keras_aug:
            X, y = next(keras_aug.flow(X, y, batch_size=batch_size))
        yield (X, y)


def load_img(img_path, target_shape, padding=None, my_aug=False):
    """Loads image"""

    if target_shape[2] == 1:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    else:
        img = cv2.imread(img_path)
    if padding == 'ruling_gray':
        border = ruling_gray(img)
    else:
        border = (0, 0, 0)
    img = resize_with_padding(img, target_shape, border)
    if my_aug:
        img = random_augmentation(img, border)
    return img


def random_augmentation(img, border):
    augs = ('rotation', 'translation', 'zoom')
    aug = random.choice(augs)
    if aug == 'rotation':
        angle = random.randint(-360, 360)
        img = rotate(img, angle, border_value=border)
    elif aug == 'translation':
        h, w = img.shape[:2]
        if h > w:
            x = random.randint(-20, 20)
            y = 0
        else:
            x = 0
            y = random.randint(-20, 20)
        img = translate(img, x, y, border_value=border)
    elif aug == 'zoom':
        amount = round(random.uniform(0.8, 1.2), 2)
        img = zoom(img, amount, border_value=border)
    flip_value = random.randint(-1, 2)
    if flip_value != 2:
        # -1:x&y, 0:x, 1:y, 2: no flip
        img = cv2.flip(img, flip_value)
    return img


def ruling_gray(img):
    hist = cv2.calcHist(img, [0], None, [256], [0, 256])
    b = g = r = int(np.argmax(hist))
    return b, g, r


def average_color(img):
    avg_per_row = np.average(img, axis=0)
    avg_colors = np.average(avg_per_row, axis=0).astype(int)
    # Convert numpy int to normal int, opencv throws error otherwise.
    avg_colors = [c.item() for c in avg_colors]
    b, g, r = avg_colors
    return b, g, r


def resize_with_padding(img, target_shape, padding=(0, 0, 0),
                       interpolation=cv2.INTER_LINEAR):
    """Resize image while maintaining aspect ratio"""

    target_h, target_w, _ = target_shape
    h, w = img.shape[:2]
    if h > w:
        r = target_h / float(h)
        dim = (int(w * r), target_h)
    else:
        r = target_w / float(w)
        dim = (target_w, int(h * r))
    img = cv2.resize(img, dim, interpolation=interpolation)
    h, w = img.shape[:2]
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    pad_top = pad_h // 2
    pad_bot = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    b, g, r = padding
    img = cv2.copyMakeBorder(img, pad_top, pad_bot, pad_left, pad_right,
                             borderType=cv2.BORDER_CONSTANT,
                             value=[b, g, r])
    return img


def gray2color(img):
    return np.stack([img] * 3, axis=-1)


def bgr2rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def translate(img, x, y, border_value=(0, 0, 0)):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             borderValue=border_value)
    return shifted


def rotate(img, angle, center=None, scale=1.0, border_value=(0, 0, 0)):
    h, w = img.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (w, h), borderValue=border_value)
    return rotated


def resize(img, width=None, height=None, inter=cv2.INTER_AREA):
    if not width and not height:
        return img
    dim = None
    h, w = img.shape[:2]
    if not width:
        r = height / float(h)
        dim = (int(w * r), height)
    if not height:
        r = width / float(w)
        dim = (width, int(h * r))
    resized = cv2.resize(img, dim, interpolation=inter)
    return resized


def zoom(img, amount=1, border_value=(0, 0, 0)):
    zoomed = cv2.resize(img, None, fx=amount, fy=amount, 
                        interpolation=cv2.INTER_LINEAR)
    zh, zw = zoomed.shape[:2]
    h, w = img.shape[:2]
    if amount < 1:
        x1 = int((w-zw)/2)
        x2 = w - zw - x1
        zoomed = cv2.copyMakeBorder(zoomed, x1, x2, x1, x2, 
                                    borderType=cv2.BORDER_CONSTANT,
                                    value=border_value)
    else:
        x1 = (zw-w)/2
        x2 = zw-x1
        x1 = int(x1)
        x2 = int(x2)
        zoomed = zoomed[x1:x2, x1:x2]
    return zoomed


def plot_img(img, title='insert title here'):#, cmap='BGR'):
    plt.axis('off')
    plt.title(title)
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    elif img.shape[2] == 1:
        img = img.reshape(img.shape[0], img.shape[1])
        plt.imshow(img, cmap='gray')
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    plt.show()


def plot_history(history, outpath=None, separate=False):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    if separate:
        plt.plot(epochs, acc, 'bo', label='Training accuracy')
        plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
        plt.title('Training and validation accuracy') 
        plt.xlabel('Epoch')
        plt.legend()
        plt.figure()
        plt.plot(epochs, loss, 'bo', color='m', label='Training loss')
        plt.plot(epochs, val_loss, 'b', color='m', label='Validation loss')
        plt.title('Training and validation loss') 
        plt.xlabel('Epoch')
        plt.legend()
    else:
        plt.plot(epochs, acc, 'b', label='Training accuracy')
        plt.plot(epochs, val_acc, 'bo', label='Validation accuracy')
        plt.plot(epochs, loss, 'b', color='m', label='Training loss')
        plt.plot(epochs, val_loss, 'bo', color='m', label='Validation loss')
        plt.title('Training and validation progress')
        plt.xlabel('Epoch')
        plt.legend()
    if outpath:
        plt.savefig(outpath)
    else:
        plt.show()


def plot_histogram(image, title, mask=None):
    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    plt.figure()
    plt.title(title)
    plt.xlabel('Bins')
    plt.ylabel('Pixel Count')

    for (chan, color) in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], mask, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlim([0, 256])
    plt.show()
