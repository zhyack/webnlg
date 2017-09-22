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
                try:
                    guess = chardet.detect(s)
                    return unicode(s, guess["encoding"])
                except:
                    return s
def _2utf8(s):
    return _2uni(s).encode('UTF-8')
def _2gbk(s):
    return _2uni(s).encode('GBK')

def catUNI(s, c):
    return s+_2uni(c)
def catUTF8(s, c):
    return _2utf8(_2uni(s)+_2uni(c))
def catGBK(s, c):
    return _2gbk(_2uni(s)+_2uni(c))

def loadDict(pf):
    f = open(pf, 'r')
    lcnt = 0
    ret = dict()
    r_ret = dict()
    for l in f.readlines():
        [w, _] = l.split()
        w = _2uni(w)
        ret[w]=lcnt
        r_ret[lcnt]=w
        lcnt += 1
    return ret, r_ret

def processing(pf, d):
    fin = open(pf,'r')
    fout = open(pf+'.post','w')
    linecnt = 0
    for line in fin.readlines():
        linecnt += 1
        line = _2uni(line).split()
        outline = ''
        i = 0
        while(i < len(line)):
            word = line[i]
            if word not in d:
                good = False
                print _2gbk(word)
                for l in range(2):
                    for r in range(5):
                        if i-l>=0 and i+r<len(line) and ''.join(line[i-l:i+r+1] in d):
                            if l>0:
                                outline+=''.join(line[i-l:i+r+1])
                            else:
                                outline+=' '+''.join(line[i-l:i+r+1])
                            i += r+1
                            good = True
                            print _2gbk(outline)
                            break
                    if good:
                        break
                if not good:
                    raise Exception("@ Line %d..."%(linecnt))
            else:
                outline+=' '+line[i]
                i+=1
        outline = _2utf8(outline[1:])
        fout.write(outline+'\n')
    fin.close()
    fout.close()

dict_dst, _ = loadDict('dict_dst_all')
processing('model_3_predictions_dev_delex.txt', dict_dst)
processing('model_3_predictions_train_2_delex.txt', dict_dst)
