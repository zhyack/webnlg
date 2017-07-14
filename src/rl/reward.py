from data_utils import *
from data_utils import _2uni, _2utf8, _2gbk
import os

def runMayGetValue(command_s, outputParser = None):
    outputs = os.popen(command_s)
    if outputParser==None:
        try:
            return float(outputs[0].strip())
        except TypeError,ValueError:
            return None
    return outputParser(outputs)

def bleuPerlParser(sl):
    ret = None
    for s in sl:
        if s.find('BLEU')!=-1 and s.find('BP')!=-1 and s.find('ratio')!=-1 and s.find('hyp_len')!=-1 and s.find('ref_len')!=-1:
            st = s.find('=')+1
            en = s.find(',', st)
            ret = float(s[st:en])
            break
    return ret


def bleuPerlInstance():
    command_s = 'cd ../../tools/baseline; python webnlg_relexicalise.py -i ../../data/ -f ../../src/rl/data/predictions.txt'
    runMayGetValue(command_s)
    command_s = 'cd ../../tools/; ./multi-bleu.perl  baseline/all-notdelex-reference0.lex baseline/all-notdelex-reference1.lex  baseline/all-notdelex-reference2.lex baseline/all-notdelex-reference3.lex  baseline/all-notdelex-reference4.lex baseline/all-notdelex-reference5.lex  baseline/all-notdelex-reference6.lex baseline/all-notdelex-reference7.lex < baseline/relexicalised_predictions.txt'
    p = bleuPerlParser
    return runMayGetValue(command_s, p)
