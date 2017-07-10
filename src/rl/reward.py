from data_utils import *
from data_utils import _2uni, _2utf8, _2gbk

import os

def runAndGetValue(command_s, outputParser = None):
    outputs = os.popen(command_s)
    if outputParser==None:
        try:
            return float(outputs[0].strip())
        except ValueError:
            return None
    return outputParser(outputs)

def bleuPerlParser()

command_s = './multi-bleu.perl data/all-notdelex-reference0.lex data/all-notdelex-reference1.lex data/all-notdelex-reference2.lex data/all-notdelex-reference3.lex data/all-notdelex-reference4.lex data/all-notdelex-reference5.lex data/all-notdelex-reference6.lex data/all-notdelex-reference7.lex < data/relexicalised_predictions.txt'
