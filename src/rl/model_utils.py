from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import numpy as np
from six.moves import range

from data_utils import *
import random

import seq2seq_model

saver = tf.train.Saver(tf.global_variables(), max_to_keep=50)

def loadModelFromFolder(sess, pf):
    config = json2load(pf+'/config.json')
    ret = seq2seq_model.Seq2SeqModel(config)
    sess.run(ret.init_model)
    saver.restore(sess, pf+"/checkpoint")
    print("Restored model from %s"%pf)
    return ret, config

def saveModelToFolder(sess, pf, config):
    save2json(config, pf+'/config.json')
    saver.save(sess, pf+'checkpoint', global_step=n_iter)
    print("Model saved at %s"%(pf+'checkpoint-'+str(n_iter)))

def instanceOfInitModel(sess, config):
    ret = seq2seq_model.Seq2SeqModel(config)
    sess.run(ret.init_model)
    print('Model Initialized.')
    return ret
