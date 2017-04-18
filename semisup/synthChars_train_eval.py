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
    using Synthetic Characters (Chars74K) dataset.

#100/300, 20/100 per class -> 24 / 17.7 after 2k5
#100/300 20/100 per class -> 18.2 after 2k5, flat
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import tensorflow as tf
import semisup
import numpy as np
# np.set_printoptions(threshold=np.inf)
import time
from tensorflow.contrib import slim
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import timeline
from tensorflow.python.lib.io import file_io

from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tools.tree import findLabelsFromTree, getWalkerLabel
tf.logging.set_verbosity(tf.logging.INFO)

FLAGS = flags.FLAGS

flags.DEFINE_integer('sup_per_class', 20,
                     'Number of labeled samples used per class.')

flags.DEFINE_integer('sup_seed', -1,
                     'Integer random seed used for labeled set selection.')

flags.DEFINE_integer('sup_batch_size', 100,
                     'Number of labeled samples per batch.')

flags.DEFINE_integer('unsup_batch_size', 300,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_integer('train_depth', 1,
                     'Max depth of tree to train')

flags.DEFINE_integer('eval_interval', 500,
                     'Number of steps between evaluations.')

flags.DEFINE_float('learning_rate', 1e-3, 'Initial learning rate.')

flags.DEFINE_float('decay_factor', 0.33, 'Learning rate decay factor.')

flags.DEFINE_float('decay_steps', 5000,
                   'Learning rate decay interval in steps.')

flags.DEFINE_float('visit_weight', 1.0, 'Weight for visit loss.')

flags.DEFINE_integer('max_steps', 20000, 'Number of training steps.')

flags.DEFINE_string('logdir', '/tmp/semisup_', 'Training log path.')

from tools import synthChars as char_tools

IMAGE_SHAPE = char_tools.IMAGE_SHAPE


def main(_):
  # TODO supervised, unsupervised and test images could overlap (if very unlucky)
  train_images, train_labels, train_data_labels, tree = char_tools.get_data('train', FLAGS.sup_per_class, seed=1)
  train_images_unsup, train_images_unsup_labels, _, _ = char_tools.get_data('train', 100, seed=2)
  test_images, test_labels, test_data_labels, _ = char_tools.get_data('test', 20, seed=3)

  # Sample labeled training subset.
  seed = FLAGS.sup_seed if FLAGS.sup_seed != -1 else None

  graph = tf.Graph()
  with graph.as_default():
    model = semisup.SemisupModel(semisup.architectures.mnist_model, tree.num_labels,
                                 IMAGE_SHAPE, treeStructure=tree, maxDepth=FLAGS.train_depth)

    # Set up inputs.
    t_unsup_images, _ = semisup.create_input(train_images_unsup, train_images_unsup_labels,
                                             batch_size=FLAGS.unsup_batch_size)
    t_sup_images, t_sup_labels = semisup.create_input(train_images, train_labels,
                                                      FLAGS.sup_batch_size)
    # Compute embeddings and logits.
    t_sup_emb = model.image_to_embedding(t_sup_images)
    t_unsup_emb = model.image_to_embedding(t_unsup_images)
    t_sup_logit = model.embedding_to_logit(t_sup_emb)

    # Add losses.
    model.add_tree_semisup_loss(
      t_sup_emb, t_unsup_emb, t_sup_labels, visit_weight=FLAGS.visit_weight)
    model.add_tree_logit_loss(t_sup_logit, t_sup_labels, weight=0.2)

    t_learning_rate = tf.train.exponential_decay(
      FLAGS.learning_rate,
      model.step,
      FLAGS.decay_steps,
      FLAGS.decay_factor,
      staircase=True)
    train_op = model.create_train_op(t_learning_rate)
    summary_op = tf.summary.merge_all()

    def evaluate_test_set(sess):

      nodeLabels = classify(model, train_images, tree, sess)
      walkerNodeLabels = np.asarray([getWalkerLabel(label, tree.depth, tree.num_nodes)
                                     for label in train_labels])

      for i in range(FLAGS.train_depth):
        printConfusionMatrix(walkerNodeLabels[:, i], nodeLabels[:, i],
                             tree.level_sizes[i + 1], "train dimension " + str(i))

      nodeLabels = classify(model, test_images, tree, sess)
      walkerNodeLabels = np.asarray([getWalkerLabel(label, tree.depth, tree.num_nodes)
                                     for label in test_labels])

      for i in range(FLAGS.train_depth):
        printConfusionMatrix(walkerNodeLabels[:, i], nodeLabels[:, i],
                             tree.level_sizes[i + 1], "test dimension " + str(i))

        # test_summary = tf.Summary(
        #   value=[tf.Summary.Value(
        #       tag='Test Err', simple_value=test_err)])

        # summary_writer.add_summary(summaries, step)
        # summary_writer.add_summary(test_summary, step)

        # saver.save(sess, FLAGS.logdir, model.step)

    # override function from slim.learning to include test set evaluation
    def train_step(session, *args, **kwargs):
      total_loss, should_stop = slim.learning.train_step(session, *args, **kwargs)

      if (train_step.step+1) % FLAGS.eval_interval == 0 or train_step.step == 2:
        print('Step: %d' % train_step.step)
        evaluate_test_set(session)

      train_step.step += 1
      return total_loss, should_stop

    train_step.step = 0

    slim.learning.train(
      train_op,
      train_step_fn=train_step,
      logdir=FLAGS.logdir,
      summary_op=summary_op,
      init_fn=None,
      number_of_steps=FLAGS.max_steps,
      save_summaries_secs=300,
      save_interval_secs=600)


def classify(model, images, tree, sess):
  pred = model.classify(images, sess)

  res = []

  for i in range(pred.shape[0]):
    nodes, _ = findLabelsFromTree(tree, pred[i, :])

    res = res + [nodes]

  return np.asarray(res)


def printConfusionMatrix(train_labels, test_pred, num_labels, name=""):

  conf_mtx = semisup.confusion_matrix(train_labels, test_pred, num_labels)
  test_err = (train_labels != test_pred).mean() * 100

  print(conf_mtx)
  print(name + ' error: %.2f %%' % test_err)
  print()


if __name__ == '__main__':
  app.run()
