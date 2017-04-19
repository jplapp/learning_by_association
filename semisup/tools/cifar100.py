"""

Definitions and utilities for the CIFAR 100 model.

This file contains functions that are needed for semisup training and evalutaion
on the Cifar 100 dataset.
The tree is defined by the superclass > class structure in the dataset.

They are used in cifar100_train_eval.py.

"""

from __future__ import division
from __future__ import print_function

import pickle
from tools import data_dirs
from os import path
from tools.tree import *

DATADIR = data_dirs.cifar100

IMAGE_SHAPE = [32, 32, 3]



nodes = []
for i in range(20):
  node = TreeNode("superclass " + str(i), leafs=range(i * 5, i * 5 + 5))
  nodes = nodes + [node]

root = TreeNode("root", children=nodes)

#for testing: tree with only leafs
#root = TreeNode("root", leafs=range(100))

tree = TreeStructure(root)


def unpickle(file):
  with open(path.join(DATADIR, file), 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')
  return dict


def get_data_from_python(name, num_per_class=10, seed=47):
  """Get a split from the cifar 100 dataset, from the original binary version

  Args:
   name: 'train' or 'test' 
   num: How many samples to read (randomly) from each class of data set

  Returns:
   images, labels
  """
  data = unpickle(name)  # name either train or test

  labels = np.asarray(data[b'fine_labels'])
  images = data[b'data']

  np.random.seed(seed)

  selectedImages = np.zeros([num_per_class*100, IMAGE_SHAPE[0],IMAGE_SHAPE[1], IMAGE_SHAPE[2]])
  selectedLabels = []

  for label in range(100):
    imgIndices = np.where(labels == label)[0]

    indices = np.random.choice(imgIndices, min(len(imgIndices), num_per_class), False)

    chosenImages = images[indices, :]

    # consecutive 1024 entries store color channels of 32x32 image
    # -> we have to reshape manually
    for i in range(num_per_class):
      img_flat = chosenImages[i, :]
      img_R = img_flat[0:1024].reshape((32, 32))
      img_G = img_flat[1024:2048].reshape((32, 32))
      img_B = img_flat[2048:3072].reshape((32, 32))
      img = np.dstack((img_R, img_G, img_B))
      selectedImages[label*num_per_class + i, :,:,:] = img

    treeNode = tree.lookupMap[label]
    selectedLabels = selectedLabels + ([treeNode.getLabels()] * num_per_class)

  labels = np.asarray(selectedLabels)

  selectedImages = selectedImages.reshape([selectedImages.shape[0],
                                           IMAGE_SHAPE[0], IMAGE_SHAPE[1],IMAGE_SHAPE[2]])

  print("loaded cifar dataset", selectedImages.shape, labels.shape)
  return [selectedImages, labels, tree]

def get_data(name, num_per_class=10, seed=47):
  """Get a split from the cifar 100 dataset, from the preprocessed numpy version

  Args:
   name: 'train' or 'test' 
   num: How many samples to read (randomly) from each class of data set

  Returns:
   images, labels
  """
  data = np.load(path.join(DATADIR, 'cifar100.npy'), encoding='latin1')[()]

  if name is 'test':
    labels = data['testLabels']
    images = data['testData']
  else:
    labels = data['trainLabels']
    images = data['trainData']

  labels = np.asarray(labels)

  np.random.seed(seed)

  selectedImages = np.zeros([num_per_class*100, IMAGE_SHAPE[0],IMAGE_SHAPE[1], IMAGE_SHAPE[2]])
  selectedLabels = []

  for label in range(100):
    imgIndices = np.where(labels == (label+1))[0]  #labels are from 1-100 here, we'll treat them as 0-99

    indices = np.random.choice(imgIndices, min(len(imgIndices), num_per_class), False)

    chosenImages = images[indices, :]

    for i in range(num_per_class):
      img_flat = chosenImages[i, :]
      img_R = img_flat[0]
      img_G = img_flat[1]
      img_B = img_flat[2]
      img = np.dstack((img_R, img_G, img_B))
      selectedImages[label*num_per_class + i, :,:,:] = img

    treeNode = tree.lookupMap[label]
    selectedLabels = selectedLabels + ([treeNode.getLabels()] * num_per_class)

  labels = np.asarray(selectedLabels)

  print("loaded cifar dataset", selectedImages.shape, labels.shape)
  return [selectedImages, labels, tree]


#res = get_data('train')
#a = 2


