from data_utils import *
from data_utils import _2uni, _2utf8, _2gbk
import os

def runMayGetValue(command_s, outputParser = None):
    outputs = os.popen(command_s)
    if outputParser==None:
        try:
            return float(outputs[0].strip())
        except TypeError,ValueError:
            print outputs
            return None
    return outputParser(outputs)

def bleuPerlParser(sl):
    ret = None
    print sl
    for s in sl:
        if s.find('BLEU')!=-1 and s.find('BP')!=-1 and s.find('ratio')!=-1 and s.find('hyp_len')!=-1 and s.find('ref_len')!=-1:
            st = s.find('=')+1
            en = s.find(',', st)
            ret = float(s[st:en])
            break
    return ret


def bleuPerlInstance():
    command_s = 'cd ../data_utils/webnlg-baseline; python webnlg_relexicalise.py -i ../../../data/ -f ../../rl/data/predictions.txt'
    runMayGetValue(command_s)
    command_s = 'cd ../data_utils/webnlg-baseline; perl multi-bleu.perl  all-notdelex-reference0.lex all-notdelex-reference1.lex  all-notdelex-reference2.lex all-notdelex-reference3.lex  all-notdelex-reference4.lex all-notdelex-reference5.lex  all-notdelex-reference6.lex all-notdelex-reference7.lex < relexicalised_predictions.txt'
    p = bleuPerlParser
    return runMayGetValue(command_s, p)

def contentPenalty(inputs, outputs, rev_dict_src, dict_dst):
    print(inputs.shape)
    print(outputs.shape)
    batch_size = len(inputs)
    assert(batch_size == len(outputs))
    max_len = outputs.shape[1]
    ret = []
    score_board = [0]*len(dict_dst)
    for ind_src in rev_dict_src:
        word = rev_dict_src[ind_src]
        if word.upper()==word and dict_dst.has_key(word):
            ind_dst = dict_dst[word]
            score_board[ind_dst]=-1.0
    for i in range(batch_size):
        sb = copy.deepcopy(score_board)
        for ind in inputs[i]:
            ind_src = int(ind)
            word = rev_dict_src[ind_src]
            if word.upper()==word and word!='<PAD>' and dict_dst.has_key(word):
                ind_dst = dict_dst[word]
                sb[ind_dst]=1.0
        ret.append([])
        predictions = outputs[i].argmax(axis=-1)
        for j in range(max_len):
            r = sb[int(predictions[j])]
            ret[i].append(sb)
            if r > 0:
                sb[int(predictions[j])] -= 0.5
    return np.array(ret, dtype=np.float32)
