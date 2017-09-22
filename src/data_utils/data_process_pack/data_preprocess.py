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
#for train_delex, generate reference
scr_refs = defaultdict(list)
for src, trg in zip(train_triple_delex, train_lex_delex):
    scr_refs[src].append(trg)
# length of the value with max elements
max_refs = sorted(scr_refs.values(), key=len)[-1]
keys = [key for (key, value) in sorted(scr_refs.items())]
values = [value for (key, value) in sorted(scr_refs.items())]
# write the source file not delex
with open('train-delex-non-repeat-triple.txt', 'w+') as f:
    f.write('\n'.join(keys))
# write references files
for j in range(0, len(max_refs)):
    with open('./ref/train-delex-non-repeat-reference' + str(j) + '.lex', 'w+') as f:
        out = ''
        for ref in values:
            try:
                out += ref[j] + '\n'
            except:
                out += '\n'
        f.write(out)

# dev_triple_delex = pickle.load(open("./source_out_dev_delex.txt", "rb"))
# dev_lex_delex = pickle.load(open("./target_out_dev_delex.txt", "rb"))
#
#
# train_triple_relex = pickle.load(open("./source_out_train_relex.txt", "rb"))
# train_lex_relex = pickle.load(open("./target_out_train_relex.txt", "rb"))
#
#
# dev_triple_relex = pickle.load(open("./source_out_dev_relex.txt", "rb"))
# dev_lex_relex = pickle.load(open("./target_out_dev_relex.txt", "rb"))
#
#
# train_rplc_list = pickle.load(open("./rplc_list_train.txt", "rb"))
# dev_rplc_list = pickle.load(open("./rplc_list_dev.txt", "rb"))


# train_data_number_all = len(train_triple_relex)
# index_list = []
# for i in range(0, train_data_number_all):
#     index_list.append(i)
# random.seed(20)
# random.shuffle(index_list)

# train_1_index = []
# for i in range(0, int(0.9 * train_data_number_all)):
#     train_1_index.append(index_list[i])
# train_1_index.sort()
#
# print (len(index_list))
# print (len(train_1_index))
#
# train_2_index = []
# index_list = []
# for i in range(0, train_data_number_all):
#     index_list.append(i)
# for i in index_list:
#     if i not in train_1_index:
#         train_2_index.append(i)
#
#
# pickle.dump(train_1_index, open("./train_1_index.txt", "wb"))
# pickle.dump(train_2_index, open("./train_2_index.txt", "wb"))
#
#
# train_1_triple_relex = [train_triple_relex[i] for i in train_1_index]
# train_1_triple_delex = [train_triple_delex[i] for i in train_1_index]
# train_1_lex_relex = [train_lex_relex[i] for i in train_1_index]
# train_1_lex_delex = [train_lex_delex[i] for i in train_1_index]
#
# train_2_triple_relex = [train_triple_relex[i] for i in train_2_index]
# train_2_triple_delex = [train_triple_delex[i] for i in train_2_index]
# train_2_lex_relex = [train_lex_relex[i] for i in train_2_index]
# train_2_lex_delex = [train_lex_delex[i] for i in train_2_index]



# create reference for datas

#
# #for dev_relex, generate reference
# scr_refs = defaultdict(list)
# for src, trg in zip(dev_triple_relex, dev_lex_relex):
#     scr_refs[src].append(trg)
# # length of the value with max elements
# max_refs = sorted(scr_refs.values(), key=len)[-1]
# keys = [key for (key, value) in sorted(scr_refs.items())]
# values = [value for (key, value) in sorted(scr_refs.items())]
# # write the source file not delex
# print (len(keys))
# with open('dev-relex-non-repeat-triple.txt', 'w+') as f:
#     f.write('\n'.join(keys))
# # write references files
# for j in range(0, len(max_refs)):
#     with open('./ref/dev-relex-non-repeat-reference' + str(j) + '.lex', 'w+') as f:
#         out = ''
#         for ref in values:
#             try:
#                 out += ref[j] + '\n'
#             except:
#                 out += '\n'
#         f.write(out)
#
#
# #for train_1_relex, generate reference
# scr_refs = defaultdict(list)
# for src, trg in zip(train_1_triple_relex, train_1_lex_relex):
#     scr_refs[src].append(trg)
# # length of the value with max elements
# max_refs = sorted(scr_refs.values(), key=len)[-1]
# keys = [key for (key, value) in sorted(scr_refs.items())]
# values = [value for (key, value) in sorted(scr_refs.items())]
# # write the source file not delex
# with open('train_1-relex-non-repeat-triple.txt', 'w+') as f:
#     f.write('\n'.join(keys))
# print (len(keys))
# # write references files
# for j in range(0, len(max_refs)):
#     with open('./ref/train_1-relex-non-repeat-reference' + str(j) + '.lex', 'w+') as f:
#         out = ''
#         for ref in values:
#             try:
#                 out += ref[j] + '\n'
#             except:
#                 out += '\n'
#         f.write(out)
#
#
# #for train_2_relex, generate reference
# scr_refs = defaultdict(list)
# for src, trg in zip(train_2_triple_relex, train_2_lex_relex):
#     scr_refs[src].append(trg)
# # length of the value with max elements
# max_refs = sorted(scr_refs.values(), key=len)[-1]
# keys = [key for (key, value) in sorted(scr_refs.items())]
# values = [value for (key, value) in sorted(scr_refs.items())]
# # write the source file not delex
# print (len(keys))
# with open('train_2-relex-non-repeat-triple.txt', 'w+') as f:
#     f.write('\n'.join(keys))
# # write references files
# for j in range(0, len(max_refs)):
#     with open('./ref/train_2-relex-non-repeat-reference' + str(j) + '.lex', 'w+') as f:
#         out = ''
#         for ref in values:
#             try:
#                 out += ref[j] + '\n'
#             except:
#                 out += '\n'
#         f.write(out)
#
#
# #for dev_delex, generate reference
# scr_refs = defaultdict(list)
# for src, trg in zip(dev_triple_delex, dev_lex_delex):
#     scr_refs[src].append(trg)
# # length of the value with max elements
# max_refs = sorted(scr_refs.values(), key=len)[-1]
# keys = [key for (key, value) in sorted(scr_refs.items())]
# values = [value for (key, value) in sorted(scr_refs.items())]
# # write the source file not delex
# with open('dev-delex-non-repeat-triple.txt', 'w+') as f:
#     f.write('\n'.join(keys))
# # write references files
# for j in range(0, len(max_refs)):
#     with open('./ref/dev-delex-non-repeat-reference' + str(j) + '.lex', 'w+') as f:
#         out = ''
#         for ref in values:
#             try:
#                 out += ref[j] + '\n'
#             except:
#                 out += '\n'
#         f.write(out)
#

# #for train_1_delex, generate reference
# scr_refs = defaultdict(list)
# for src, trg in zip(train_1_triple_delex, train_1_lex_delex):
#     scr_refs[src].append(trg)
# # length of the value with max elements
# max_refs = sorted(scr_refs.values(), key=len)[-1]
# keys = [key for (key, value) in sorted(scr_refs.items())]
# values = [value for (key, value) in sorted(scr_refs.items())]
# # write the source file not delex
# with open('train_1-delex-non-repeat-triple.txt', 'w+') as f:
#     f.write('\n'.join(keys))
# # write references files
# for j in range(0, len(max_refs)):
#     with open('./ref/train_1-delex-non-repeat-reference' + str(j) + '.lex', 'w+') as f:
#         out = ''
#         for ref in values:
#             try:
#                 out += ref[j] + '\n'
#             except:
#                 out += '\n'
#         f.write(out)
#
#
# #for train_2_delex, generate reference
# scr_refs = defaultdict(list)
# for src, trg in zip(train_2_triple_delex, train_2_lex_delex):
#     scr_refs[src].append(trg)
# # length of the value with max elements
# max_refs = sorted(scr_refs.values(), key=len)[-1]
# keys = [key for (key, value) in sorted(scr_refs.items())]
# values = [value for (key, value) in sorted(scr_refs.items())]
# # write the source file not delex
# with open('train_2-delex-non-repeat-triple.txt', 'w+') as f:
#     f.write('\n'.join(keys))
# # write references files
# for j in range(0, len(max_refs)):
#     with open('./ref/train_2-delex-non-repeat-reference' + str(j) + '.lex', 'w+') as f:
#         out = ''
#         for ref in values:
#             try:
#                 out += ref[j] + '\n'
#             except:
#                 out += '\n'
#         f.write(out)
#
# dev_relex = open("./dev-relex-non-repeat-triple.txt", "r").read().split("\n")
# index = []
# for i in dev_relex:
#     if i not in dev_triple_relex:
#         print ("wraasdfasd")
#     else:
#         index.append(str(dev_triple_relex.index(i)))
# open("./dev_relex_no_repeat_index.txt", "w").write("\n".join(index))
#
# dev_delex = open("./dev-delex-non-repeat-triple.txt", "r").read().split("\n")
# index = []
# for i in dev_triple_delex:
#     if i not in dev_delex:
#         print ("sdfasdf")
#     else:
#         index.append(str(dev_delex.index(i)))
# open("./dev_delex_no_repeat_index.txt", "w").write("\n".join(index))
