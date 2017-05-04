#! /usr/bin/env python
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

Association-based semi-supervised hierarchical training example 
    using CIFAR100 dataset.

depth 1, cifarnet
no prep, 500/0 b128 l1e-3 ds2000 -> 60% on test after 10000steps
no prep, 200/300 b64/128 l1e-3 ds4000 vw 0.1 -> 70% on test after 10000steps
no prep, 100/500 b64/128 l1e-3 ds5000 vw 0.2 -> 76% on test after 50000steps

depth 2, cifarnet
no prep, 500/0 b64 l1e-3 ds5000 -> 67%,70% on test after 12000steps
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import semisup
import numpy as np
from random import randint
from tensorflow.contrib import slim

from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tools.cifar100 import tree
from tools.tree import getWalkerLabel, findLabelsFromTreeMultitask
from tools import cifar100 as cifar_tools, dataset_factory, preprocessing_factory, cifar100, data_dirs

tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = flags.FLAGS

# flags.DEFINE_integer('sup_per_class', 500,
#                      'Number of labeled samples used per class.')
#
# flags.DEFINE_integer('unsup_per_class', 500,
#                      'Number of labeled samples used per class.')
#
# flags.DEFINE_integer('test_per_class', 20,
#                      'Number of labeled samples used per class.')

flags.DEFINE_integer('sup_batch_size', 128,
                     'Number of labeled samples per batch.')

flags.DEFINE_integer('unsup_batch_size', 128,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_bool('unsup', False, 'Add unsupervised training samples')

flags.DEFINE_integer('train_logit_depth', 1,
                     'Max depth of logits to use to train tree')

flags.DEFINE_integer('train_walker_depth', 1,
                     'Max depth of walker loss in the tree')

flags.DEFINE_integer('eval_interval', 1000,
                     'Number of steps between evaluations.')

flags.DEFINE_float('learning_rate',0.001, 'Initial learning rate.')

flags.DEFINE_float('decay_factor', 0.1, 'Learning rate decay factor.')

flags.DEFINE_float('decay_steps', 15000,
                   'Learning rate decay interval in steps.')

flags.DEFINE_float('visit_weight', 0.2, 'Weight for visit loss.')

flags.DEFINE_float('walker_weight', 0.5, 'Weight for walker loss.')

flags.DEFINE_float('gpu_fraction', 1.0, 'Fraction of GPU to use.')

flags.DEFINE_integer('max_steps', 50000, 'Number of training steps.')

flags.DEFINE_string('logdir','/data/logs', 'Training log path.')
flags.DEFINE_bool('randomize_logdir', False, 'Whether to add a random string to the logdir to make it unique')
flags.DEFINE_string('dataset_dir', data_dirs.cifar100, 'Dataset Location.')
flags.DEFINE_bool('log_losses', False, 'Log losses during training')

IMAGE_SHAPE = cifar_tools.IMAGE_SHAPE


def main(_):
  train_depth = max(FLAGS.train_logit_depth, FLAGS.train_walker_depth)

  graph = tf.Graph()
  with graph.as_default():


    preprocessing_name = "cifarnet"#FLAGS.preprocessing_name or FLAGS.model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
      preprocessing_name,
      is_training=True)

    if not FLAGS.dataset_dir:
      raise ValueError('You must supply the dataset directory with --dataset_dir')

    with tf.device('/device:CPU:0'):  #preprocessing on cpu is apparently way faster
      trainDatasetSup = dataset_factory.get_dataset(
        'cifar100', 'train_sup', FLAGS.dataset_dir)

      trainProviderSup = slim.dataset_data_provider.DatasetDataProvider(
        trainDatasetSup,
        num_readers=4,  # FLAGS.num_readers,
        common_queue_capacity=10 * FLAGS.sup_batch_size,
        common_queue_min=5 * FLAGS.sup_batch_size)
      [train_images_sup, train_labels_sup] = trainProviderSup.get(['image', 'label'])

      train_image_size = 32

      train_images_sup = image_preprocessing_fn(train_images_sup, train_image_size, train_image_size)

      trainDatasetUnsup = dataset_factory.get_dataset(
        'cifar100', 'train_unsup', FLAGS.dataset_dir)

      trainProviderUnsup = slim.dataset_data_provider.DatasetDataProvider(
        trainDatasetUnsup,
        num_readers=4,  # FLAGS.num_readers,
        common_queue_capacity=10 * FLAGS.unsup_batch_size,
        common_queue_min=5 * FLAGS.unsup_batch_size)
      [train_images_unsup, train_labels_unsup] = trainProviderUnsup.get(['image', 'label'])

      train_images_unsup = image_preprocessing_fn(train_images_unsup, train_image_size, train_image_size)


      testDataset = dataset_factory.get_dataset(
        'cifar100', 'test', FLAGS.dataset_dir)
      testProvider = slim.dataset_data_provider.DatasetDataProvider(
        testDataset,
        num_readers=2,  # FLAGS.num_readers,
        common_queue_capacity=4 * FLAGS.sup_batch_size,
        common_queue_min=2 * FLAGS.sup_batch_size)
      [test_images, test_labels] = testProvider.get(['image', 'label'])
      test_images = image_preprocessing_fn(test_images, train_image_size, train_image_size)

    model = semisup.SemisupModel(semisup.architectures.cifar_model, tree.num_labels,
                                 IMAGE_SHAPE, treeStructure=tree,
                                 maxLogitDepth=FLAGS.train_logit_depth,
                                 maxWalkerDepth=FLAGS.train_walker_depth)

    # Set up inputs.
    if FLAGS.unsup:
      t_unsup_images, _ = semisup.create_input(train_images_unsup, train_labels_unsup,
                                               batch_size=FLAGS.unsup_batch_size)
    t_sup_images, t_sup_labels = semisup.create_input(train_images_sup, train_labels_sup,
                                                      FLAGS.sup_batch_size)

    # Compute embeddings and logits.
    t_sup_emb = model.image_to_embedding(t_sup_images)
    if FLAGS.unsup: t_unsup_emb = model.image_to_embedding(t_unsup_images)
    t_sup_logit = model.embedding_to_logit(t_sup_emb)

    # Add losses.
    if FLAGS.unsup:
      model.add_tree_semisup_loss(
        t_sup_emb, t_unsup_emb, t_sup_labels,
        walker_weight=FLAGS.walker_weight, visit_weight=FLAGS.visit_weight)
    model.add_tree_multitask_logit_loss(t_sup_logit, t_sup_labels, weight=1.)

    logdir = FLAGS.logdir
    if FLAGS.randomize_logdir:
      logdir = logdir + str(randint(0, 99999))

    t_learning_rate = tf.train.exponential_decay(
      FLAGS.learning_rate,
      model.step,
      FLAGS.decay_steps,
      FLAGS.decay_factor,
      staircase=True)
    train_op = model.create_train_op(t_learning_rate)
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(logdir, graph)

    train_scores = [[] for _ in range(train_depth)]
    test_scores = [[] for _ in range(train_depth)]

    def evaluate_test_set(sess, global_step):

      ti, gtLabels = model.materializeTensors([train_images_sup, train_labels_sup], 5000, sess)

      nodeLabels = classify(model, ti, tree, sess)
      for i in range(train_depth):
        err = printConfusionMatrix(gtLabels[:, i], nodeLabels[:, i],
                                   tree.level_sizes[i + 1], "train dimension " + str(i))
        train_scores[i] = train_scores[i] + [err]

      #todo make sure this gets all test images
      ti, gtLabels = model.materializeTensors([test_images, test_labels], 10000, sess)
      nodeLabels = classify(model, ti, tree, sess)

      for i in range(train_depth):
        err = printConfusionMatrix(gtLabels[:, i], nodeLabels[:, i],
                                   tree.level_sizes[i + 1], "test dimension " + str(i))
        test_scores[i] = test_scores[i] + [err]

        test_summary = tf.Summary(
            value=[tf.Summary.Value(
                tag='Test error level '+str(i), simple_value=err)])

        summary_writer.add_summary(test_summary, global_step)

    # override function from slim.learning to include test set evaluation
    def train_step(session, train_op, global_step, *args, **kwargs):
      total_loss, should_stop = slim.learning.train_step(session, train_op, global_step, *args, **kwargs)

      step = session.run(global_step)
      if (step+1) % FLAGS.eval_interval == 0 or step == 9:
        print('Step: %d' % step)
        evaluate_test_set(session, step)

      return total_loss, should_stop


    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=FLAGS.gpu_fraction)
    
    slim.learning.train(
      train_op,
      train_step_fn=train_step,
      logdir=logdir,
      summary_op=summary_op,
      session_config=tf.ConfigProto(gpu_options=gpu_options),#device_count={'GPU': 0}
      number_of_steps=FLAGS.max_steps,
      save_summaries_secs=300,
      summary_writer=summary_writer,
      log_every_n_steps=100,
      #trace_every_n_steps=10,
      save_interval_secs=300)


    print('train accuracies', train_scores)
    print('test accuracies', test_scores)
    print('flags', FLAGS.__flags)


def classify(model, images, tree, sess):
  pred = model.classify(images, sess)

  res = []

  for i in range(pred.shape[0]):
    nodes = findLabelsFromTreeMultitask(tree, pred[i, :])

    res = res + [nodes]

  return np.asarray(res)


def printConfusionMatrix(train_labels, test_pred, num_labels, name=""):

  conf_mtx = semisup.confusion_matrix(train_labels, test_pred, num_labels)
  test_err = (train_labels != test_pred).mean() * 100

  print(conf_mtx)
  print(name + ' error: %.2f %%' % test_err)
  return test_err


if __name__ == '__main__':
  app.run()
