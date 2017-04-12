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
# np.set_printoptions(threshold=np.inf)

from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tools.tree import findLabelsFromTree, getWalkerLabel

FLAGS = flags.FLAGS

flags.DEFINE_integer('sup_per_class', 500,
                     'Number of labeled samples used per class.')

flags.DEFINE_integer('unsup_per_class', 500,
                     'Number of labeled samples used per class.')

flags.DEFINE_integer('test_per_class', 20,
                     'Number of labeled samples used per class.')

flags.DEFINE_integer('sup_batch_size', 128,
                     'Number of labeled samples per batch.')

flags.DEFINE_integer('unsup_batch_size', 128,
                     'Number of unlabeled samples per batch.')

flags.DEFINE_integer('train_depth', 1,
                     'Max depth of tree to train')

flags.DEFINE_integer('eval_interval', 500,
                     'Number of steps between evaluations.')

flags.DEFINE_float('learning_rate', 5e-3, 'Initial learning rate.')

flags.DEFINE_float('decay_factor', 0.33, 'Learning rate decay factor.')

flags.DEFINE_float('decay_steps', 5000,
                   'Learning rate decay interval in steps.')

flags.DEFINE_float('visit_weight', 0.2, 'Weight for visit loss.')

flags.DEFINE_integer('max_steps', 100, 'Number of training steps.')

flags.DEFINE_string('logdir', '/tmp/semisup_', 'Training log path.')
flags.DEFINE_bool('log_losses', False, 'Log losses during training')

from tools import cifar100 as cifar_tools

IMAGE_SHAPE = cifar_tools.IMAGE_SHAPE


def main(_):
  train_images, train_labels, tree = cifar_tools.get_data('train', FLAGS.sup_per_class, seed=1)
  train_images_unsup, train_images_unsup_labels, _ = cifar_tools.get_data('train', FLAGS.unsup_per_class, seed=2)
  test_images, test_labels, _ = cifar_tools.get_data('test', FLAGS.test_per_class, seed=3)

  graph = tf.Graph()
  with graph.as_default():
    model = semisup.SemisupModel(semisup.architectures.cifar_model, tree.num_labels,
                                 IMAGE_SHAPE, treeStructure=tree, maxDepth=FLAGS.train_depth)

    # Set up inputs.
    #t_unsup_images, _ = semisup.create_input(train_images_unsup, train_images_unsup_labels,
    #                                         batch_size=FLAGS.unsup_batch_size)
    t_sup_images, t_sup_labels = semisup.create_input(train_images, train_labels,
                                                      FLAGS.sup_batch_size)
    # Compute embeddings and logits.
    t_sup_emb = model.image_to_embedding(t_sup_images)
    #t_unsup_emb = model.image_to_embedding(t_unsup_images)
    t_sup_logit = model.embedding_to_logit(t_sup_emb)

    # Add losses.
    #model.add_tree_semisup_loss(
    #  t_sup_emb, t_unsup_emb, t_sup_labels, visit_weight=FLAGS.visit_weight)
    model.add_tree_logit_loss(t_sup_logit, t_sup_labels, weight=1.)

    t_learning_rate = tf.train.exponential_decay(
      FLAGS.learning_rate,
      model.step,
      FLAGS.decay_steps,
      FLAGS.decay_factor,
      staircase=True)
    train_op = model.create_train_op(t_learning_rate)
    summary_op = tf.summary.merge_all()

    #summary_writer = tf.summary.FileWriter(FLAGS.logdir, graph)

    #saver = tf.train.Saver()
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)

  train_scores = [[] for _ in range(FLAGS.train_depth)]
  test_scores = [[] for _ in range(FLAGS.train_depth)]

  with tf.Session(graph=graph, config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
    tf.global_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(FLAGS.max_steps):
      _, summaries, *losses = sess.run([train_op, summary_op] + model.logit_losses + model.walker_losses + model.visit_losses)
      if FLAGS.log_losses: print("losses:", losses)
      if (step + 1) % FLAGS.eval_interval == 0 or step == 99:
        print('Step: %d' % step)

        nodeLabels = classify(model, train_images, tree)
        walkerNodeLabels = np.asarray([getWalkerLabel(label, tree.depth, tree.num_nodes)
                                       for label in train_labels])

        for i in range(FLAGS.train_depth):
          err = printConfusionMatrix(walkerNodeLabels[:, i], nodeLabels[:, i],
                               tree.level_sizes[i + 1], "train dimension " + str(i))
          train_scores[i] = train_scores[i] + [err]


        nodeLabels = classify(model, test_images, tree)
        walkerNodeLabels = np.asarray([getWalkerLabel(label, tree.depth, tree.num_nodes)
                                       for label in test_labels])

        for i in range(FLAGS.train_depth):
          err = printConfusionMatrix(walkerNodeLabels[:, i], nodeLabels[:, i],
                               tree.level_sizes[i + 1], "test dimension " + str(i))
          test_scores[i] = test_scores[i] + [err]


        # test_summary = tf.Summary(
        #   value=[tf.Summary.Value(
        #       tag='Test Err', simple_value=test_err)])

        # summary_writer.add_summary(summaries, step)
        # summary_writer.add_summary(test_summary, step)

        # saver.save(sess, FLAGS.logdir, model.step)

    coord.request_stop()
    coord.join(threads)

    print('train accuracies', train_scores)
    print('test accuracies', test_scores)


def classify(model, images, tree):
  pred = model.classify(images)

  res = []

  for i in range(pred.shape[0]):
    nodes, _ = findLabelsFromTree(tree, pred[i, :])

    res = res + [nodes]

  return np.asarray(res)


def printConfusionMatrix(train_labels, test_pred, num_labels, name=""):

  #conf_mtx = semisup.confusion_matrix(train_labels, test_pred, num_labels)
  test_err = (train_labels != test_pred).mean() * 100

  #print(conf_mtx)
  print(name + ' error: %.2f %%' % test_err)
  return test_err


if __name__ == '__main__':
  app.run()
