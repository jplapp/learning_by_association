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

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import semisup
import numpy as np
#np.set_printoptions(threshold=np.inf)

from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from tools.group import findLabelFromTree

FLAGS = flags.FLAGS

flags.DEFINE_integer('sup_per_class', 10,
                     'Number of labeled samples used per class.')

flags.DEFINE_integer('sup_seed', -1,
                     'Integer random seed used for labeled set selection.')

flags.DEFINE_integer('sup_batch_size', 50,
                     'Number of labeled samples per batch.')

flags.DEFINE_integer('unsup_batch_size', 300,
                     'Number of unlabeled samples per batch.')

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

NUM_LABELS = char_tools.NUM_LABELS
IMAGE_SHAPE = char_tools.IMAGE_SHAPE


def main(_):
  # TODO supervised, unsupervised and test images could overlap (if very unlucky)
  train_images, train_labels, _, tree = char_tools.get_data('train', FLAGS.sup_per_class, seed=1)
  train_images_unsup, train_images_unsup_labels,_,_ = char_tools.get_data('train', 30, seed=2)
  test_images, test_labels, test_data_labels, _ = char_tools.get_data('test', 10, seed=3)

  # Sample labeled training subset.
  seed = FLAGS.sup_seed if FLAGS.sup_seed != -1 else None

  graph = tf.Graph()
  with graph.as_default():
    model = semisup.SemisupModel(semisup.architectures.mnist_model, tree.num_labels,
                                 IMAGE_SHAPE, treeStructure=tree)

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
    model.add_tree_logit_loss(t_sup_logit, t_sup_labels)

    t_learning_rate = tf.train.exponential_decay(
        FLAGS.learning_rate,
        model.step,
        FLAGS.decay_steps,
        FLAGS.decay_factor,
        staircase=True)
    train_op = model.create_train_op(t_learning_rate)
    summary_op = tf.summary.merge_all()

    summary_writer = tf.summary.FileWriter(FLAGS.logdir, graph)

    saver = tf.train.Saver()

  with tf.Session(graph=graph) as sess:
    tf.global_variables_initializer().run()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    for step in range(FLAGS.max_steps):
      _, summaries = sess.run([train_op, summary_op])
      if (step + 1) % FLAGS.eval_interval == 0 or step == 99:
        print('Step: %d' % step)
        pred = model.classify(test_images)
        test_pred = [0] * pred.shape[0]

        for i in range(pred.shape[0]):
          test_pred[i] = findLabelFromTree(tree, pred[i,:])

        test_pred = np.asarray(test_pred)
        conf_mtx = semisup.confusion_matrix(test_data_labels, test_pred, NUM_LABELS)
        test_err = (test_data_labels != test_pred).mean() * 100
        print(conf_mtx)
        print('Test error: %.2f %%' % test_err)
        print()

        test_summary = tf.Summary(
            value=[tf.Summary.Value(
                tag='Test Err', simple_value=test_err)])

        summary_writer.add_summary(summaries, step)
        summary_writer.add_summary(test_summary, step)

        saver.save(sess, FLAGS.logdir, model.step)

    coord.request_stop()
    coord.join(threads)


if __name__ == '__main__':
  app.run()
