from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import chardet
def _2uni(s):
    try:
        return unicode(s)
    except:
        try:
            return unicode(s, 'UTF-8')
        except UnicodeDecodeError:
            try:
                return unicode(s, 'GBK')
            except UnicodeDecodeError:
                guess = chardet.detect(s)
                try:
                    return unicode(s, guess["encoding"])
                except:
                    return ""
def _2utf8(s):
    return _2uni(s).encode('UTF-8')
def _2gbk(s):
    return _2uni(s).encode('GBK')

def dict2utf8(d):
    ret = dict()
    for k in d.keys():
        nk = k
        nv = d[k]
        if isinstance(k, basestring):
            nk = _2utf8(k)
        if isinstance(nv, basestring):
            ret[nk]=_2utf8(nv)
        elif isinstance(nv, list):
            ret[nk]=list2utf8(nv)
        elif isinstance(nv, dict):
            ret[nk]=dict2utf8(nv)
        else:
            ret[nk]=nv
    return ret
def list2utf8(l):
    ret = []
    for item in l:
        nv = item
        if isinstance(nv, basestring):
            ret.append(_2utf8(nv))
        elif isinstance(nv, list):
            ret.append(list2utf8(nv))
        elif isinstance(nv, dict):
            ret.append(dict2utf8(nv))
        else:
            ret.append(nv)
    return ret
import json
def save2json(d, pf):
    f = open(pf,'w')
    f.write(json.dumps(d, ensure_ascii=False, indent=4))
    f.close()
def json2load(pf):
    f = open(pf,'r')
    s = ''.join(f.readlines())
    d = json.loads(s)
    d = dict2utf8(d)
    f.close()
    return d

def loadDict(pf):
    f = open(pf, 'r')
    lcnt = 0
    ret = dict()
    r_ret = []
    for l in f.readlines():
        [w, _] = l.split()
        ret[w]=lcnt
        r_ret.append(w)
        lcnt += 1
    if not ret.has_key('<bOS>'):
        ret['<BOS>']=lcnt
        r_ret.append('<BOS>')
        lcnt += 1
    if not ret.has_key('<EOS>'):
        ret['<EOS>']=lcnt
        r_ret.append('<EOS>')
        lcnt += 1
    if not ret.has_key('<UNK>'):
        ret['<UNK>']=lcnt
        r_ret.append('<UNK>')
        lcnt += 1
    if not ret.has_key('<PAD>'):
        ret['<PAD>']=lcnt
        r_ret.append('<PAD>')
        lcnt += 1
    return ret, r_ret

def arrangeBuckets(seq_pairs, buckets):
    ret = []
    n_bs = len(buckets)
    for _ in range(n_bs):
        ret.append([])
    for sp in seq_pairs:
        n_sp = len(sp)
        ok_d = False
        for b in range(n_bs):
            assert(n_sp == len(buckets[b]))
            ok_l = True
            for s in range(n_sp):
                if len(sp[s].split()) >= buckets[b][s]-1:
                    ok_l = False
                    break
            if ok_l:
                ret[b].append(sp)
                ok_d = True
                break
        if not ok_d:
            ret[-1].append(sp)
    return ret

def npShuffle(npls):
    indices = np.arange(len(npls[0]))
    np.random.shuffle(indices)
    for i in range(len(npls)):
        npls[i] = npls[i][indices]
    return npls

def dataSeq2Onehot(s, full_dict, max_len):
    ret = []
    ndict = len(full_dict)
    assert(full_dict.has_key('<UNK>'))
    assert(full_dict.has_key('<BOS>'))
    assert(full_dict.has_key('<EOS>'))
    assert(full_dict.has_key('<PAD>'))
    nr = 0
    ret.append([0]*ndict)
    ret[-1][full_dict['<BOS>']]=1
    nr += 1
    for w in s.split():
        if not full_dict.has_key(w):
            w = '<UNK>'
        ret.append([0]*ndict)
        ret[-1][full_dict[w]]=1
    nr = len(ret)
    if nr > max_len-1:
        print(_2utf8('Len of setence %d ||| %s ||| exceed... Clipping...'%(nr, s)))
        ret = ret[:max_len-1]
        nr = max_len-1
    ret.append([0]*ndict)
    ret[-1][full_dict['<EOS>']]=1
    nr += 1
    while(nr < max_len):
        ret.append([0]*ndict)
        ret[-1][full_dict['<PAD>']]=1
        nr += 1
    return ret

def dataSeqs2Digits(s, full_dict, max_len=None):
    ret = []
    ndict = len(full_dict)
    assert(full_dict.has_key('<UNK>'))
    assert(full_dict.has_key('<BOS>'))
    assert(full_dict.has_key('<EOS>'))
    assert(full_dict.has_key('<PAD>'))
    ret.append(full_dict['<BOS>'])
    for w in s.split():
        if not full_dict.has_key(w):
            w = '<UNK>'
        ret.append(full_dict[w])
    nr = len(ret)
    if max_len==None:
        max_len=nr+1
    if nr > max_len-1:
        print(_2utf8('Len of setence %d ||| %s ||| exceed... Clipping...'%(nr, s)))
        ret = ret[:max_len-1]
        nr = max_len-1
    ret.append(full_dict['<EOS>'])
    nr += 1
    while(nr < max_len):
        ret.append(full_dict['<PAD>'])
        nr += 1
    return ret

def dataSeqs2NpSeqs(seqs, full_dict, max_len, dtype=np.int32, shuffled=False, subf=dataSeqs2Digits):
    ret = []
    for s in seqs:
        x = subf(s, full_dict, buckets)
        ret.append(x)
    ret_len = []
    for s in ret:
        ret_len.append(len(ret))
    ret = np.array(ret, dtype=dtype)
    ret_len = np.array(ret, dtype=np.int32)
    if shuffled:
        [ret] = npShuffle([ret])
    return ret, ret_len

def dataLogits2Seq(x, full_dict, calc_argmax=False):
    if calc_argmax:
            x = x.argmax(axis=-1)
    return ' '.join(full_dict[x] for x in x)
