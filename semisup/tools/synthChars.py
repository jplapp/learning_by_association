"""
Copyright 2016 Google Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Definitions and utilities for the Synthetic Characters model.

This file contains functions that are needed for semisup training and evalutaion
on the Synthetic Characters dataset.
They are used in synthChars_train_eval.py.

"""

from __future__ import division
from __future__ import print_function

import numpy as np
from tools import data_dirs

DATADIR = data_dirs.synthChars


NUM_LABELS = 62 # 62 + 3 TODO implement tree
IMAGE_SHAPE = [28, 28, 1]  #originally 128x128, downsampled

#!/usr/bin/python
from os import listdir, path
from PIL import Image as PImage

def loadImages(label, num_samples):
    # return array of num_samples images

    imagesList = listdir(label)
    np.random.seed(47)  # TODO make configurable
    indices = np.random.choice(len(imagesList), min(len(imagesList), num_samples), False)
    imagesList = [ imagesList[i] for i in indices]

    loadedImages = []
    for image in imagesList:
        img = PImage.open(path.join(label, image))
        img = img.resize([IMAGE_SHAPE[0], IMAGE_SHAPE[1]])
        img = np.asarray(img, dtype=np.uint8).T  # transpose

        loadedImages.append(img)

    return loadedImages


def get_data(name, num_per_class=10):
  """Get a split from the synth dataset.

  Args:
   name: 'train' or 'test' TODO is currently ignore, train and test might overlap
   num: How many samples to read (randomly) from each class of data set

  Returns:
   images, labels
  """

  labelsList = listdir(DATADIR)

  imList = []
  labelList = []

  labelIndex = 0

  for label in labelsList:
    imgs = loadImages(path.join(DATADIR, label), num_per_class)
    imList = imList + imgs
    labelList = labelList + ([labelIndex] * len(imgs))
    labelIndex = labelIndex + 1

  labels = np.asarray(labelList)
  allImgs = np.asarray(imList)
  allImgs = allImgs.reshape([allImgs.shape[0], allImgs.shape[1], allImgs.shape[2], 1])

  print(allImgs.shape, labels.shape)
  return [allImgs, labels]


get_data('train', 10)