score_list = open('./score_model.txt', 'r').read().split('\n')
score_list = [[(float)(j) for j in (i.split(" "))] for i in score_list[:-1]]
open('./score_model.txt', 'w').write('')
model_number = 2
id = 0
#score_list = [float(i) for i in score_list[:-1]]
for i in range(0, len(score_list[0])):
    list_tmp = []
    for j in range(0, model_number):
        list_tmp.append(score_list[j][i])
    flag = 0
    for k in list_tmp:
        if k != 0:
            flag = 1
    if flag == 1:
        id = list_tmp.index(max(list_tmp))
        break
    else:
        print "Attention!"
        continue
    
print "score_list"
print score_list
str_out = open("./source_for_bleu_" + str(id) + ".txt", "r").read()
line_number = int(open("./line_number.txt",'r').read())
print str_out
open("./output_upper.txt", "a").write(str_out + "\n")
print open("./output_upper.txt", "r").read()