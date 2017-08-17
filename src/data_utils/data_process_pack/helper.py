import os
import random
import re
import json
import sys
import getopt
from collections import defaultdict
import pickle

train_triple_delex = pickle.load(open("./source_out_train_delex.txt", "rb"))
train_lex_delex = pickle.load(open("./target_out_train_delex.txt", "rb"))

dev_triple_delex = pickle.load(open("./source_out_dev_delex.txt", "rb"))
dev_lex_delex = pickle.load(open("./target_out_dev_delex.txt", "rb"))


train_triple_relex = pickle.load(open("./source_out_train_relex.txt", "rb"))
train_lex_relex = pickle.load(open("./target_out_train_relex.txt", "rb"))


dev_triple_relex = pickle.load(open("./source_out_dev_relex.txt", "rb"))
dev_lex_relex = pickle.load(open("./target_out_dev_relex.txt", "rb"))


train_rplc_list = pickle.load(open("./rplc_list_train.txt", "rb"))
dev_rplc_list = pickle.load(open("./rplc_list_dev.txt", "rb"))


train_1_index = pickle.load(open("./train_1_index.txt", "rb"))
train_2_index = pickle.load(open("./train_2_index.txt", "rb"))


train_1_triple_relex = [train_triple_relex[i] for i in train_1_index]
train_1_triple_delex = [train_triple_delex[i] for i in train_1_index]
train_1_lex_relex = [train_lex_relex[i] for i in train_1_index]
train_1_lex_delex = [train_lex_delex[i] for i in train_1_index]

train_2_triple_relex = [train_triple_relex[i] for i in train_2_index]
train_2_triple_delex = [train_triple_delex[i] for i in train_2_index]
train_2_lex_relex = [train_lex_relex[i] for i in train_2_index]
train_2_lex_delex = [train_lex_delex[i] for i in train_2_index]


def relex_train_1(text, index):
    '''use the index in train_1 to relex'''
    index_in_train = train_1_index[index]
    rplc_dict = train_rplc_list[index_in_train]
    for key in sorted(rplc_dict):
        text = text.replace(key + ' ', rplc_dict[key] + ' ')
    return text

def relex_train_2(text, index):
    '''use the index in train_2 to relex'''
    index_in_train = train_2_index[index]
    rplc_dict = train_rplc_list[index_in_train]
    for key in sorted(rplc_dict):
        text = text.replace(key + ' ', rplc_dict[key] + ' ')
    return text


def relex_dev(text, index):
    '''use the index in dev to relex'''
    rplc_dict = dev_rplc_list[index]
    for key in sorted(rplc_dict):
        text = text.replace(key + ' ', rplc_dict[key] + ' ')
    return text

def export_data(triple, lex, prefix='noname'):
    def _export_(data, fn):
        f = open(fn, 'w')
        f.write('\n'.join(data))
        f.close()
    _export_(triple, prefix+'.triple')
    _export_(lex, prefix+'.lex')

print(len(train_1_triple_delex),len(train_1_lex_delex))
export_data(train_1_triple_delex, train_1_lex_delex, 'train-1-webnlg-all-delex')
export_data(train_2_triple_delex, train_2_lex_delex, 'train-2-webnlg-all-delex')
export_data(dev_triple_delex, dev_lex_delex, 'dev-webnlg-all-delex')
