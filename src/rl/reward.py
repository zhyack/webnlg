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

def contentPenalty(inputs, outputs, rev_dict_src, dict_dst, targets):
    # print(inputs.shape)
    # print(outputs.shape)
    batch_size = len(inputs)
    assert(batch_size == len(outputs))
    max_len = outputs.shape[1]
    ret = []
    all_keys = [False]*len(dict_dst)
    for ind_src in rev_dict_src:
        word = rev_dict_src[ind_src]
        if word.upper()==word and len(word)>2 and word!='<EOS>' and dict_dst.has_key(word):
            ind_dst = dict_dst[word]
            all_keys[ind_dst]=True
    for i in range(batch_size):
        # sb = copy.deepcopy(score_board)
        poskey_cnt = 0
        pos_keys = [0]*len(dict_dst)
        for ind in inputs[i]:
            ind_src = int(ind)
            word = rev_dict_src[ind_src]
            if word.upper()==word and len(word)>2 and word!='<PAD>' and word!='<EOS>' and dict_dst.has_key(word):
                ind_dst = dict_dst[word]
                if pos_keys[ind_dst]!=2:
                    poskey_cnt += 1
                pos_keys[ind_dst]=2
        ret.append([])
        predictions = outputs[i].argmax(axis=-1)
        for j in range(max_len):
            score_board = [0.0]*len(dict_dst)
            p = int(predictions[j])
            if (all_keys[p] and pos_keys[p]==0):
                score_board[p] = 0
            elif (all_keys[p] and pos_keys[p]==2):
                score_board[p] = 2.0
                pos_keys[p] -= 1
            elif (all_keys[p] and pos_keys[p]==1):
                score_board[p] = 1.0
                pos_keys[p] -= 1
            else:
                score_board[p] = 1.0

            if j>0 and predictions[j]==predictions[j-1]:
                score_board[p] -= 1.0
            ret[i].append(score_board)
        remain_poskey_cnt = 0
        for k in range(len(dict_dst)):
            if pos_keys[k] == 2:
                remain_poskey_cnt += 1
        remain_penalty = remain_poskey_cnt*-0.5/poskey_cnt
        for j in range(max_len):
            p = int(predictions[j])
            if ret[i][j][p] != 2.0:
                ret[i][j][p] += remain_penalty
                ret[i][j][p] = max(0.0,ret[i][j][p])
    return np.array(ret, dtype=np.float32)