#coding=utf-8
import os
import random
import re
import json
import sys
import getopt
from collections import defaultdict
import pickle

test_rplc_list = json.load(open("./test_rplc_dict.json"))

input_file = open("./prediction_test_delex.txt").read().split("\n")

out = ""
for index, text in enumerate(input_file):
    rplc_dict = test_rplc_list[index]
    for key in sorted(rplc_dict):
        text = text.replace(key + ' ', rplc_dict[key] + ' ')
    out += text + "\n"

open("./prediction_test_relex.txt", "w").write(out)