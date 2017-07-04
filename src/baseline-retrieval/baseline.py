import math
import sys

base = r'data\base.txt'
test = r'data\test.txt'


def distance(u, v):
    s = set()
    for n in u:
        s.add(n)
    for n in v:
        s.add(n)                            # s     是两个向量中所有维度的集合

    sum = 0
    for n in s:
        if n in u:
            ux = u[n]
        else:
            ux = 0
        if n in v:
            vx = v[n]
        else:
            vx = 0
        sum += (ux - vx) ** 2

    return sum


if __name__ == '__main__':
    # 从语料库base中读取句子 整合到corpus这个list中
    # 若有多个base可以整合到一起
    # 需要满足句子没有标点符号以及含有标点符号的缩写(可以预处理解决)
    handle = open(base, 'r')
    corp = handle.readlines()
    num_sent = len(corp)
    corpus = []
    num_word = dict()                       # num_word  所有单词出现次数的dict
    words = []                              # words     每一句话拆开成单词的list
    all_word = 0                            # all_word  语料库单词总数
    for co in corp:
        co = co.strip()
        corpus.append(co)
        co = co.split(' ')
        words.append(co)
        all_word += len(co)
        for word in co:
            if word in num_word:
                num_word[word] += 1
            else:
                num_word[word] = 1
    # print(words)
    # print(num_word)

    all_vec = []
    for i in range(num_sent):
        len_sent_i = len(words[i])
        dic = dict()
        for j in range(len_sent_i):
            if words[i][j] in dic:
                dic[words[i][j]] += 1
            else:
                dic[words[i][j]] = 1
        tf = dict()
        for n in dic:
            tf[n] = dic[n] / len_sent_i                     # tf
        idf = dict()
        for n in dic:
            idf[n] = tf[n] * math.log(all_word / (1 + num_word[n]), 2)   # tf-idf
        # print(idf)
        all_vec.append(idf)
    handle.close()

    # print(all_vec)                          # all_vec   所有句子的向量

    # 接下来要从test中读取验证数据
    ft = open(test, 'r')
    line = ft.readline().strip()
    while line:
        # 计算line的向量 并和其他向量比较距离
        dic = dict()
        line_words = line.split(' ')        # line_words    读进来的一个句子分成多个单词
        line_word_num = len(line_words)     # line_word_num 一个句子总共有多少个单词
        for line_word in line_words:
            if line_word in dic:
                dic[line_word] += 1
            else:
                dic[line_word] = 1          # dic           句子单词和词数的字典

        tf = dict()
        for n in dic:
            tf[n] = dic[n] / line_word_num
        idf = dict()
        for n in dic:
            if n in num_word:
                quot = 1 + num_word[n]
            else:
                quot = 1
            idf[n] = tf[n] * math.log(all_word / quot, 2)       # idf   是输入的句子的代表向量

        min_dis = sys.maxsize
        tag = 0
        min_tag = 0
        # print(idf)
        for vec in all_vec:
            dis = distance(vec, idf)
            # print('the distance between %s and %s is %f' % (corp[tag], line, dis))
            if dis < min_dis:
                min_dis = dis
                min_tag = tag
            tag += 1

        # print(min_dis)
        print(corp[min_tag])

        line = ft.readline().strip()

    ft.close()
