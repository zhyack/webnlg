#coding: utf-8
import os
import json
import yaml
import xml.etree.ElementTree as et
from nltk.tokenize import word_tokenize
import nltk
import copy
def _2uni(s):
    try:
        return s.decode('GBK')
    except UnicodeDecodeError:
        try:
            return s.decode('UTF-8')
        except UnicodeDecodeError:
            return s
def _2gbk(s):
    try:
        return s.encode('GBK')
    except UnicodeError:
        try:
            return s.decode('UTF-8').encode('GBK')
        except UnicodeDecodeError:
            return s.decode('GBK').encode('GBK')
def _2utf(s):
    try:
        return s.encode('UTF-8')
    except UnicodeError:
        try:
            return s.decode('GBK').encode('UTF-8')
        except UnicodeDecodeError:
            try:
                return s.decode('UTF-8').encode('UTF-8')
            except UnicodeError:
                return ""
def queryS(q):
    return _2utf(raw_input(_2gbk(q)).strip().rstrip())
def queryL(q):
    return queryS(q).split()
def queryLS(q):
    s = queryS(q)
    return s, s.split()
def printS(s):
    s = _2gbk(s)
    print s
def printL(l):
    for i in len(l):
        l[i] = str(l[i])
    printS(' '.join(l))
def save2json(d, pf):
    f = open(pf,'w')
    s = json.dumps(d, ensure_ascii=False, indent=4).split('\n')
    for line in s:
        f.write(line+'\n')
    f.close()
def json2load(pf):
    f = open(pf,'r')
    s = ''.join(f.readlines())
    f.close()
    def custom_str_constructor(loader, node):
        return loader.construct_scalar(node).encode('utf-8')
    yaml.add_constructor(u'tag:yaml.org,2002:str', custom_str_constructor)
    return yaml.load(s)
def xml2dict(pf, isfile=True):
    def getNodeDict(node):
        ret = node.attrib
        if len(_2utf(node.text.strip().rstrip())):
            ret['text'] = _2utf(node.text.strip().rstrip())
        for cn in node:
            ck = cn.tag
            if not ret.has_key(ck):
                ret[ck]=[]
            ret[ck].append(getNodeDict(cn))
        return ret
    tree = None
    ret = {}
    if isfile:
        tree = et.parse(pf)
    else:
        tree = et.fromstring(pf)
    root = tree.getroot()
    ret[root.tag] = [getNodeDict(root)]
    return ret
def getData(pf):
    def trackTriple(triples, category, o_or_m='otriple'):
        def getTextInfo(s):
            l = s.split('|')
            assert(len(l)==3)
            for i in range(3):
                l[i] = l[i].strip().rstrip()
            return l[0], l[1], l[2]
        rmp = dict()
        ret = dict()
        for triple in triples[o_or_m]:
            text = triple['text']
            t1, t2, t3 = getTextInfo(text)
            if t1==t3:
                continue
            rmp[t3] = t2
        root = []
        for triple in triples[o_or_m]:
            text = triple['text']
            t1, t2, t3 = getTextInfo(text)
            if t1==t3:
                continue
            if not rmp.has_key(t1):
                root.append(t1)
                rmp[t1]='...'
        if len(root)>1:
            print pf
            print triples
            raise Exception('ROOT ERROR')
        elif len(root)==0:
            t1, _, _ = getTextInfo(triples[o_or_m][0]['text'])
            root=[t1]
        pre = [category]
        ret[category] = root[0]
        while(len(root)):
            for triple in triples[o_or_m]:
                text = triple['text']
                t1, t2, t3 = getTextInfo(text)
                if t1==t3:
                    continue
                if t1 == root[0]:
                    t2 = pre[0]+'____'+t2
                    while ret.has_key(t2):
                        t2 += '*'
                    ret[t2]=t3
                    root.append(t3)
                    pre.append(t2)
            del(root[0])
            del(pre[0])
        return ret

    ret = {}
    ret['infos'] = []
    ret['answers'] = []
    if not os.path.isfile(pf):
        for subpf in os.listdir(pf):
            subret = getData(pf+'/'+subpf)
            ret['infos'] += subret['infos']
            ret['answers'] += subret['answers']
        return ret
    if not pf.endswith('.xml'):
        return ret
    entries = xml2dict(pf)['benchmark'][0]['entries'][0]['entry']
    for entry in entries:
        ret['infos'].append([[],[]])
        ret['answers'].append([])
        if entry.has_key('originaltripleset'):
            for originaltriple in entry['originaltripleset']:
                ret['infos'][-1][0].append(copy.deepcopy(trackTriple(originaltriple, entry['category'])))
        if entry.has_key('modifiedtripleset'):
            for modifiedtriple in entry['modifiedtripleset']:
                ret['infos'][-1][1].append(copy.deepcopy(trackTriple(modifiedtriple, entry['category'], o_or_m='mtriple')))

        if entry.has_key('lex'):
            for lex in entry['lex']:
                if lex['comment'] == 'good':
                    ret['answers'][-1].append(lex['text'])
        if len(ret['answers'][-1])==0:
            ret['answers'][-1].append('None')
    return ret
def printDataInfo(pf):
    d = None
    if os.path.isfile(pf):
        d = json2load(pf)
    else:
        d = getData(pf)
    s_key = set()
    s_pair = set()
    key_cnt = 0
    inp_cnt = 0
    ans_cnt = 0
    ind =  0
    for p in d['infos']:
        for i in range(2):
            for kd in p[i]:
                s_pair.add(';'.join(kd.keys()))
                for k in kd.keys():
                    s_key.add(k)
                    key_cnt+=1
                inp_cnt += 1
                ans_cnt += len(d['answers'][ind])
        ind += 1

    print _2gbk(pf+'\n共有%d组数据，其中：\n包含%d个键值，仅有%d种不同的键值\n包含%d个输入，有%d种不同的输入\n'%(ans_cnt, key_cnt, len(s_key), inp_cnt, len(s_pair)))
def transformData(pfsrc, pfin, pfout, only_input=False):
    data = json2load(pfsrc)
    ori_inputs = data['infos']
    ori_outputs = data['answers']
    ori_n = len(ori_inputs)
    fin = open(pfin, 'w')
    fout = open(pfout, 'w')
    fkey = open(pfin+'.key','w')
    fval = open(pfin+'.val','w')
    total_cnt = 0
    fail_cnt = 0
    for i in range(ori_n):
        inputs = ori_inputs[i][0]+ori_inputs[i][1]
        outputs = ori_outputs[i]
        for inp in inputs:
            for outp in outputs:
                (t_inp, t_outp) = copy.deepcopy((inp, outp))
                total_cnt += 1
                ok = True
                t_outp = t_outp.lower()
                kld = {}
                for k in inp.keys():
                    kld[k]=len(k)
                len_ordered_keys = sorted(kld.items(),key=lambda d:d[1], reverse=True)
                for k in inp.keys():
                    if t_inp[k].endswith('@en'):
                        t_inp[k] = t_inp[k][:-3]
                    t_inp[k] = [t_inp[k].lower()]
                    t_inp[k].append(t_inp[k][0].replace('_',' '))
                    t_inp[k].append(t_inp[k][1].replace(' ',''))
                    t_inp[k].append(t_inp[k][1][1:-1])
                    t_inp[k].append(t_inp[k][2][1:-1])
                    p = t_inp[k][1].rfind(', ')
                    t_inp[k].append(t_inp[k][1][:p]+' and '+t_inp[k][1][p+2:])
                    p = t_inp[k][2].rfind(',')
                    t_inp[k].append(t_inp[k][2][:p]+' and '+t_inp[k][2][p+1:])
                    p = t_inp[k][3].rfind(', ')
                    t_inp[k].append(t_inp[k][3][:p]+' and '+t_inp[k][3][p+2:])
                    p = t_inp[k][4].rfind(',')
                    t_inp[k].append(t_inp[k][4][:p]+' and '+t_inp[k][4][p+1:])
                    p = t_inp[k][1].rfind('(')
                    t_inp[k].append(t_inp[k][1][:p])
                    p = t_inp[k][2].rfind('(')
                    t_inp[k].append(t_inp[k][2][:p])
                    p = t_inp[k][1].rfind(' (')
                    t_inp[k].append(t_inp[k][1][:p])
                    p = t_inp[k][2].rfind(' (')
                    t_inp[k].append(t_inp[k][2][:p])
                def fuzzy_find(s, l, rs):
                    for t in l:
                        p = s.find(t)
                        if (p!=-1):
                            return p, s.replace(t, rs.replace(' ',''))
                    return -1, s
                for k in len_ordered_keys:
                    k = k[0]
                    p, t_outp = fuzzy_find(t_outp, t_inp[k], "<$%s$>"%k)
                    if (p == -1):
                        ok = False
                if only_input:
                    fin.write(' '.join(t_inp.keys())+'\n')
                    fout.write(outp+'\n')
                    for k in t_inp.keys():
                        fkey.write(k.replace(' ','')+'\n')
                        fval.write(inp[k].replace('_',' ')+'\n')
                    fkey.write('\n')
                    fval.write('\n')
                    continue
                fin.write(' '.join(t_inp.keys())+'\n')
                fout.write(t_outp+'\n')
                for k in t_inp.keys():
                    fkey.write(k.replace(' ','')+'\n')
                    fval.write(inp[k].replace('_',' ')+'\n')
                fkey.write('\n')
                fval.write('\n')
                if not ok:
                    fail_cnt += 1
                    # printS(' | '.join(inp.keys()))
                    # printS(' | '.join(inp.values()))
                    # for l in t_inp.values():
                    #     printS(' ; '.join(l))
                    # printS(outp)
                    # printS(t_outp)
    fin.close()
    fout.close()
    fval.close()
    fkey.close()
    print 'fail/total: %d/%d'%(fail_cnt, total_cnt)
def getDict(files,fdict=None):
    d,rd = [], dict()
    for pf in files:
        f = open(pf,'r')
        for line in f.readlines():
            line = _2uni(line.strip())
            l = []
            st, en = 0,0
            while(True):
                en = line.find('<$',st)
                if en == -1:
                    if st != len(line)-1:
                        l += word_tokenize(line[st:])
                    break
                if en>st:
                    l += word_tokenize(line[st:en])
                st = line.find('$>', en)+2
                l += [line[en:st]]
            for w in l:
                if not rd.has_key(w):
                    rd[w] = 1
                else:
                    rd[w]+=1
        f.close()
    dd =sorted(rd.items(),key=lambda d:d[1], reverse=True)
    dcnt = 0
    for item in dd:
        d.append(_2utf(item[0]).replace(' ',''))
        rd[item[0].replace(' ','')] = dcnt
        dcnt+=1
    if not rd.has_key('<UNK>'):
        d.append('<UNK>')
        rd['<UNK>']=dcnt
        dd.append(('<UNK>',0))
        dcnt+=1
    if fdict:
        f = open(fdict, 'w')
        for item in dd:
            f.write('%s\t%d\n'%(_2utf(item[0]).replace(' ',''),item[1]))
        f.close()
    return d,rd
def dictionarizeData(pfin, pfout, rd):
    fin = open(pfin, 'r')
    fout = open(pfout, 'w')
    for line in fin.readlines():
        line = _2uni(line.strip())
        l = []
        st, en = 0,0
        while(True):
            en = line.find('<$',st)
            if en == -1:
                if st != len(line)-1:
                    l += word_tokenize(line[st:])
                break
            if en>st:
                l += word_tokenize(line[st:en])
            st = line.find('$>', en)+2
            l += [line[en:st]]
        for w in l:
            if rd.has_key(w):
                fout.write(str(rd[w])+' ')
            else:
                fout.write(str(rd['<UNK>'])+' ')
        fout.write('\n')
    fin.close()
    fout.close()
def rebuildDict(pf):
    d = json2load(pf)
    rd = dict()
    for i in range(len(d)):
        rd[_2uni(d[i])] = i
    return d, rd

def reText(pfin, pfout, d):
    fin  = open(pfin,'r')
    fout = open(pfout,'w')
    for line in fin.readlines():
        l = line.split()
        for i in l:
            fout.write(d[int(i)]+' ')
        fout.write('\n')
    fin.close()
    fout.close()
def postProcessing(pfin, pfkey, pfval, pfout):
    fin = open(pfin, 'r')
    fout = open(pfout, 'w')
    fkey = open(pfkey, 'r')
    fval = open(pfval, 'r')
    lcnt = 0
    for in_inline in fin.readlines():
        while(True):
            k = '<$'+fkey.readline().strip()+'$>'
            v = fval.readline().strip().replace('_',' ')
            if v.endswith('@en'):
                v = v[:-3]
            if v.startswith("\"") and v.endswith("\""):
                v = v[1:-1]
            # v = _2utf(' '.join(word_tokenize(_2uni(v))))
            if k=='<$$>':
                break
            in_inline = in_inline.replace(k,v)
        fout.write(in_inline)
    fin.close()
    fout.close()
    fkey.close()
    fval.close()
# save2json(xml2dict('data/dev/3triples/3triples_Airport_dev_challenge.xml'),'tmp.json')
# save2json(getData('data/dev'),'data/dev_origin.json')
# save2json(getData('data/train'),'data/train_origin.json')
# printDataInfo('data/train')
# printDataInfo('data/dev')
# printDataInfo('data')
transformData('data/train_origin.json', 'data/train_input_text.txt', 'data/train_output_text.txt')
transformData('data/dev_origin.json', 'data/dev_input_text.txt', 'data/dev_output_text.txt')
transformData('data/train_origin.json', 'data/train_input_text.txt', 'data/train_output_text_ori.txt', only_input=True)
transformData('data/dev_origin.json', 'data/dev_input_text.txt', 'data/dev_output_text_ori.txt', only_input=True)
d1,rd1 = getDict(['data/train_input_text.txt'], 'data/dict_src')
d2,rd2 = getDict(['data/train_output_text.txt'], 'data/dict_dst')
dictionarizeData('data/train_input_text.txt', 'data/train_input_data.txt', rd1)
dictionarizeData('data/train_output_text.txt', 'data/train_output_data.txt', rd2)
dictionarizeData('data/dev_input_text.txt', 'data/dev_input_data.txt', rd1)
dictionarizeData('data/dev_output_text.txt', 'data/dev_output_data.txt', rd2)
reText('data/train_input_data.txt', 'data/train_input_text.txt', d1)
reText('data/train_output_data.txt', 'data/train_output_text.txt', d2)
reText('data/dev_input_data.txt', 'data/dev_input_text.txt', d1)
reText('data/dev_output_data.txt', 'data/dev_output_text.txt', d2)
# d1,rd1 = getDict(['data/train_input_text.txt'], 'data/dict_src')
# d2,rd2 = getDict(['data/train_output_text.txt'], 'data/dict_dst')
# postProcessing('data/predictions.txt', 'data/dev_input_text.txt.key', 'data/dev_input_text.txt.val', 'data/predict_full.txt')
