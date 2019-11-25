import random
import os
from argparse import ArgumentParser

import cv2
import numpy as np
from matplotlib import pyplot as plt

from keras.models import load_model
from keras.preprocessing import image
from keras.applications import VGG16, VGG19, DenseNet121

import file_tools
import image_tools


parser = ArgumentParser()
parser.add_argument('-m', '--model', required=True,
    help='Path to pre-trained model')
parser.add_argument('-i', '--images', required=True,
    help='Path to image folder')
parser.add_argument('-l', '--labels', required=True,
    help='Path to list of labels')
parser.add_argument('-c', '--count',
    help='How many images to predict')
parser.add_argument('-f', '--feature_extractor',
    help='Name of feature extractor: VGG16, VGG19 or DenseNet')
args = parser.parse_args()

model = load_model(args.model)
img_paths = file_tools.list_files(args.images, 'image')
if args.count:
    stop = int(args.count)
    img_paths = [p for p in img_paths]
    random.shuffle(img_paths)
    img_paths = img_paths[:stop]

results = []
class_names = []
with open(args.labels) as fh:
    for line in fh:
        class_names.append(line.strip())

if args.feature_extractor:
    target_shape = (128, 128, 3)
else:
    target_shape = (128, 128, 1)

for path in img_paths:
    img = image_tools.load_img(path, target_shape, 'ruling_gray')
    img = img.astype('float32') / 255
    img = np.reshape(img, target_shape)
    img = np.expand_dims(img, axis=0)
    if args.feature_extractor:
        if args.feature_extractor == 'VGG16':
            fe = VGG16(include_top=False, weights='imagenet', input_shape=target_shape)
        elif args.feature_extractor == 'VGG19':
            fe = VGG19(include_top=False, weights='imagenet', input_shape=target_shape)
        elif args.feature_extractor == 'DenseNet':
            fe = DenseNet121(include_top=False, weights='imagenet', input_shape=target_shape)
        else:
            raise ValueError(f"Unkown model feature extractor: {args.feature_extractor}")
        features = fe.predict(img)
        bs, d1, d2, d3 = features.shape
        features = features.reshape((bs, d1*d2*d3))
        pred = model.predict(features)
    else:
        pred = model.predict(img)
    pred = pred.argmax(axis=1)[0]
    label = class_names[pred]
    ground_truth = os.path.basename(path)
    plot = cv2.imread(path)
    image_tools.plot_img(plot, f'{label} ({ground_truth})')
