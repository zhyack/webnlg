open('./score_model.txt', 'w').write('')
line_number = int(open("./line_number.txt",'r').read())
model_number = 2
ref_max = 8
file_list = []
for i in range(0, model_number):
    file_list.append(open("./relexicalised_predictions_" + str(i) + ".txt", "r"))
    
for index, f in enumerate(file_list):
    open("./source_for_bleu_" + str(index) + ".txt", "w").write(f.read().split('\n')[line_number])
        
for i in range(0, ref_max):
    tmp_str = open("./all-notdelex-reference" + str(i) + ".lex", "r").read().split('\n')[line_number]
    open("./ref_for_bleu_" + str(i) + ".txt", "w").write(tmp_str)

open("./line_number.txt",'w').write(str(line_number + 1))