import os
import random
import re
import json
import sys
import getopt
from collections import defaultdict
import pickle

rplc_list = json.load(open("./test_rplc_dict.json"))

test_repeat_triple = open("./test-webnlg-all-delex.triple").read().split("\n")

model_number = int(open("./model_number.txt").read())

test_no_repeat = list(set(test_repeat_triple))

thr = 0.3

ratio = []

for i in range(0, model_number):
    pred = open("./model_" + str(i) +"_predictions_test_delex.txt").read().split("\n")
    tmp = []
    for index, j in enumerate(pred[:-1]):
        counter = 0.0
        for key in sorted(rplc_list[index]):
            if j.find(key) != -1:
                counter += 1.0
        tmp.append(counter / len(rplc_list[i]))
    ratio.append(tmp)

ratio = [[row[i] for row in ratio] for i in range(len(ratio[0]))]

ratio = [sum(i)/len(i) for i in ratio][:-1]

fix_index_list_repeat = []

for index, i in enumerate(ratio):
    if i < thr:
        fix_index_list_repeat.append(index)

try:
    rp_dc = json.load(open("./rp_manual.json"))
except:
    rp_dc = {}

print ("There are "  + str(len(fix_index_list_repeat)) +" lines...")
print (str(len(rp_dc)) + " of them has been manually relexed.")

need_rp = [test_repeat_triple[i] for i in fix_index_list_repeat]
need_triple = [rplc_list[i] for i in fix_index_list_repeat]
index_map =[]
for i in sorted(need_rp):
    index_map.append(need_rp.index(i))

a = input("Do you want start from the beginning?")
if a == "1":
    for in_ in index_map:
        i = need_rp[in_]
        if i in rp_dc.keys():
            continue
        print("this is the " + str(need_rp.index(i)) + "th sentence.")
        i_copy = [' ']
        for c in i:
            i_copy.append(c)
        i_copy = "".join(i_copy)
        for key in sorted(need_triple[in_]):
            i_copy = i_copy.replace(" " + key + " ", " * " + key + " * ")
        counter = 0
        new_i = []
        for char in i_copy:
            new_i.append(char)
            if char == "*":
                counter +=1
            if counter == 4:
                new_i.append("\n")
                counter = 0
        print(''.join(new_i))
        a = input("Enter your relex:")
        if a == "e":
            break
        rp_dc[i] = a
else:
    for in_ in index_map[::-1]:
        i = need_rp[in_]
        if i in rp_dc.keys():
            continue
        print("this is the " + str(need_rp.index(i)) + "th sentence.")
        i_copy = i
        for key in sorted(need_triple[in_]):
            i_copy = i_copy.replace(key + " ", " * " + key + " * ")
        counter = 0
        new_i = []
        for char in i_copy:
            new_i.append(char)
            if char == "*":
                counter +=1
            if counter == 4:
                new_i.append("\n")
                counter = 0
        print(''.join(new_i))
        a = input("Enter your relex:")
        if a == "e":
            break
        rp_dc[i] = a
json.dump(rp_dc, open("./rp_manual.json", "w"), indent=4)
