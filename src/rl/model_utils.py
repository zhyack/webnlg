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
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)
    return saver

def loadModelFromFolder(sess, saver, pf):
    config = json2load(pf+'/config.json')
    saver.restore(sess, pf+"/checkpoint")
    print("Restored model from %s"%pf)
    return ret, config

def saveModelToFolder(sess, saver, pf, config):
    save2json(config, pf+'/config.json')
    saver.save(sess, pf+'checkpoint', global_step=n_iter)
    print("Model saved at %s"%(pf+'checkpoint-'+str(n_iter)))

def instanceOfInitModel(sess, config):
    ret = seq2seq_model.Seq2SeqModel(config)
    sess.run(tf.global_variables_initializer())
    print('Model Initialized.')
    return ret


def create_learning_rate_decay_fn(decay_steps=200, decay_rate=0.01, decay_type='natural_exp_decay', start_decay_at=0, stop_decay_at=200000, min_learning_rate=None, staircase=False):

    def decay_fn(learning_rate, global_step):
        decay_type_fn = getattr(tf.train, decay_type)
        decayed_learning_rate = decay_type_fn(learning_rate=learning_rate, global_step=tf.minimum(global_step, stop_decay_at) - start_decay_at, decay_steps=decay_steps, decay_rate=decay_rate, staircase=staircase, name="decayed_learning_rate")
        final_lr = tf.train.piecewise_constant(x=global_step, boundaries=[start_decay_at], values=[learning_rate, decayed_learning_rate])

        if min_learning_rate:
            final_lr = tf.maximum(final_lr, min_learning_rate)
        return final_lr

    return decay_fn
