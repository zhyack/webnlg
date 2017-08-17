from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np
from six.moves import range

from data_utils import *
from reward import *
import random

import seq2seq_model

def initGlobalSaver():
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=200)
    return saver

def loadConfigFromFolder(config, pf):
    if os.path.isfile(pf+'/config.json'):
        config = json2load(pf+'/config.json')
    return config
def loadModelFromFolder(sess, saver, config, pf):
    if os.path.isfile(pf+'/config.json'):
        config = json2load(pf+'/config.json')
    ckpt = tf.train.get_checkpoint_state(pf)
    if ckpt!=None:
        print('Restoring checkpoint @ %s'%(ckpt.model_checkpoint_path))
        saver.restore(sess, ckpt.model_checkpoint_path)
    print("Restored model from %s"%pf)
    return config

def saveModelToFolder(sess, saver, pf, config, n_iter):
    save2json(config, pf+'/config.json')
    saver.save(sess, pf+'/checkpoint', global_step=n_iter)
    print("Model saved at %s"%(pf+'/checkpoint-'+str(n_iter)))

def instanceOfInitModel(sess, config):
    ret = seq2seq_model.Seq2SeqModel(config)
    sess.run(tf.global_variables_initializer())
    print('Model Initialized.')
    return ret


def create_learning_rate_decay_fn(decay_steps=500, decay_rate=0.7, decay_type='natural_exp_decay', start_decay_at=0, stop_decay_at=200000, min_learning_rate=0.00001, staircase=False):

    def decay_fn(learning_rate, global_step):
        decay_type_fn = getattr(tf.train, decay_type)
        decayed_learning_rate = decay_type_fn(learning_rate=learning_rate, global_step=tf.minimum(global_step, stop_decay_at) - start_decay_at, decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase, name="decayed_learning_rate")
        final_lr = tf.train.piecewise_constant(x=global_step, boundaries=[start_decay_at], values=[learning_rate, decayed_learning_rate])

        if min_learning_rate:
            final_lr = tf.maximum(final_lr, min_learning_rate)
        return final_lr

    return decay_fn
#0.3 0.95 500 800*3
# Evaluationg completed:
# Average Loss:10.916317
# BLEU:50.12
# [30.61, 41.6, 46.85, 44.41, 47.75, 45.91, 44.49, 44.15, 48.43, 48.82, 46.15, 48.49, 48.32, 48.91, 48.47, 48.32, 48.
# 99, 49.61, 49.62, 49.35, 48.78, 49.84, 48.35, 48.29, 49.28, 49.49, 50.2, 50.9, 47.45, 49.02, 49.63, 49.71, 49.59, 49.05, 49.05, 50.18, 48.8, 49.19, 49.62, 49.5, 49.95, 49.28, 50.16, 49.79, 49.7, 50.25, 49.22, 49.93, 50.09, 49.97, 50.23, 50.24, 49.65, 50.19, 49.79, 49.7, 49.11, 49.44, 49.14, 50.75, 50.36, 49.02, 50.68, 49.48, 50.23, 50.01, 50.19, 48.71, 50.23, 49.53, 49.92, 51.37, 50.25, 49.78, 50.03, 49.99, 50.51, 50.78, 49.97, 49.21, 49.36, 49.77, 49.83, 50.51, 51.25, 49.88, 49.93, 49.7, 49.61, 50.2, 50.17, 49.77, 49.97, 50.12, 49.82, 49.61, 50.21, 49.38, 49.92, 49.59, 50.02, 49.45, 49.84, 49.95, 49.66, 50.18, 49.68, 49.81, 49.6, 48.53, 50.39, 49.16, 49.42, 49.16, 49.4, 49.35, 49.64, 49.5, 50.45, 50.43, 50.2, 50.13, 49.17, 50.12, 51.01, 49.92, 48.73, 50.37, 49.48, 49.86, 50.22, 50.03, 49.95, 50.33, 50.09, 50.22, 50.44, 50.08, 49.17, 50.41, 49.64, 49.94, 49.74, 50.37, 49.33, 49.77, 50.0, 50.58, 49.88, 50.72, 49.62, 49.99, 49.77, 49.97, 49.47, 50.68, 49.26, 49.1, 50.76, 49.42, 50.12, 49.93, 50.15, 50.38, 50.13, 50.61, 49.78, 49.06, 49.82, 49.92, 49.58, 48.86, 49.32, 50.24, 49.85, 48.99, 49.68, 49.35, 49.52, 50.27, 49.8, 49.04, 50.53, 48.99, 50.33, 50.52, 49.64, 50.86, 51.03, 50.26, 50.67, 49.37, 50.1, 50.38, 50.0, 50.35, 49.88, 50.34, 49.64]


#0.3 0.9 1000 800*3
# BLEU:53.16
# [39.4, 45.81, 47.93, 48.57, 48.67, 49.59, 50.93, 50.46, 50.77, 52.17, 51.93, 51.87, 52.24, 51.78, 52.69, 53.67, 51.8
# 2, 53.11, 53.07, 52.34, 52.56, 52.83, 51.88, 53.57, 52.31, 52.5, 52.29, 53.78, 52.03, 53.28, 52.26, 53.51, 52.6, 52.58, 52.96, 53.63, 52.61, 53.3, 53.0, 52.89, 53.23, 52.28, 53.58, 52.99, 52.58, 53.09, 53.17, 52.62, 52.63, 52.25, 53.54, 53.19, 53.19, 53.71, 53.55, 53.01, 53.46, 52.99, 52.03, 53.73, 53.44, 53.44, 52.97, 54.3, 52.52, 54.05, 52.47, 53.19, 52.82, 52.59, 52.42, 52.7, 53.46, 52.48, 53.38, 53.25, 52.97, 53.69, 53.35, 53.44, 53.54, 52.02, 53.06, 53.32, 53.08, 52.93, 52.34, 52.89, 52.64, 52.87, 53.12, 53.49, 53.31, 53.38, 53.81, 54.09, 52.11, 52.54, 52.81, 52.73, 53.46, 52.86, 52.75, 53.01, 53.02, 52.97, 53.4, 53.36, 52.31, 53.47, 53.03, 53.35, 53.14, 52.79, 51.92, 53.16, 52.68, 52.71, 53.35, 53.31, 53.1, 53.1, 53.13, 53.27, 53.01, 52.72, 52.41, 52.76, 53.67, 52.24, 53.84, 53.93, 53.24, 52.58, 53.0, 53.25, 52.51, 53.83, 52.98, 53.68, 52.85, 52.38, 53.82, 52.57, 52.97, 52.9, 53.22, 52.62, 53.11, 53.41, 52.57, 52.65, 52.77, 52.89, 52.49, 52.22, 52.32, 53.35, 52.45, 52.83, 53.34, 53.71, 52.68, 53.31, 53.1, 52.89, 52.72, 53.0, 53.48, 52.53, 52.34, 53.14, 53.0, 53.25, 52.53, 52.51, 52.3, 52.71, 53.68, 53.05, 53.06, 53.48, 52.92, 52.95, 53.24, 52.87, 52.75, 53.8, 53.13, 53.03, 53.31, 52.99, 52.45, 53.71, 52.25, 53.47, 53.19, 52.82, 52.62]
