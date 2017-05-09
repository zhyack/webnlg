import re

DEV_DELEX = 'webnlg-baseline/dev-webnlg-all-delex.lex'
DEV_NOTDELEX = 'webnlg-baseline/dev-webnlg-all-notdelex.lex'

TRAIN_DELEX = 'webnlg-baseline/train-webnlg-all-delex.lex'
TRAIN_NOTDELEX = 'webnlg-baseline/train-webnlg-all-notdelex.lex'

DEV_KEY_OUTPUT = 'dev-key.txt'
DEV_VAL_OUTPUT = 'dev-val.txt'

TRAIN_KEY_OUTPUT = 'train-key.txt'
TRAIN_VAL_DELEX = 'train-val.txt'

def is_word(c):
    if ord(c) >= ord('A') and ord(c) <= ord('Z'):
        return True
    if ord(c) >= ord('a') and ord(c) <= ord('z'):
        return True
    return False

def any_word(str):
    for i in range(len(str)):
        if is_word(str[i]):
            return True
    return False

def last_big(str):
    if str == str.upper():
        return 0
    for i in range(1, len(str)):
        if str[i:] == str[i:].upper():
            return i
    return 0

def need_trans(c):
    s = set(['.', '/', '\\', '(', ')', '|', '?'])
    if c in s:
        return True
    return False

def trans_pat(str):
    res = ''
    for i in range(len(str)):
        if need_trans(str[i]):
            res = res + '\\' + str[i]
        else:
            res = res + str[i]
    return res

dfi = open(TRAIN_DELEX, 'r', encoding='utf-8')
nfi = open(TRAIN_NOTDELEX, 'r', encoding='utf-8')

dls = dfi.readlines()
nls = nfi.readlines()

dfi.close()
nfi.close()

fk = open(TRAIN_KEY_OUTPUT, 'w', encoding='utf-8')
fv = open(TRAIN_VAL_DELEX, 'w', encoding='utf-8')

n_index = 0
for dl in dls:
    dl = dl.strip()
    num_pat = 0
    l_pat = list()
    nl = nls[n_index].strip()
    n_index += 1

    group = dl.split(' ')
    pat = '^'
    pat_temp = ''
    for word in group:
        if word == word.upper() and any_word(word):
            if pat_temp == '':
                pat_temp = word
            else:
                pat_temp = pat_temp + ' ' + word
        elif any_word(word) and last_big(word) > 0:
            if pat_temp == '':
                pat_temp += word[last_big(word)]
                pat = pat + word[0:last_big(word)]
            else:
                l_pat.append(pat_temp)
                num_pat += 1
                pat = pat + '(.*) ' + word[0:last_big(word)]
                pat_temp = word[last_big(word)]

        else:
            if pat_temp == '':
                pass
            else:
                l_pat.append(pat_temp)
                num_pat += 1
                pat = pat + '(.*) '
                pat_temp = ''

            pat = pat + trans_pat(word) + ' '
            # if word == '.':
            #     pat = pat + '\\' + word + ' '
            # else:
            #     pat = pat + word + ' '
    if pat_temp == '':
        pass
    else:
        l_pat.append(pat_temp)
        num_pat += 1
        pat = pat + '(.*) '
        pat_temp = ''


    pat = pat[0:len(pat) - 1] + '$'

    m = re.match(pat, nl, re.S)

    for i in range(num_pat):
        if l_pat[i] == m.group(i + 1):
            pass
        else:
            fk.write('<$' + l_pat[i] + '$>' + '\n')
            fv.write(m.group(i+1) + '\n')

    fk.write('\n')
    fv.write('\n')


fk.close()
fv.close()









