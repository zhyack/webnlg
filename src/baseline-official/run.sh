cd /mnt/c/Users/zhy-win/Git/OpenNMT
# th preprocess.lua -train_src ../webnlg/webnlg-baseline/train-webnlg-all-delex.triple -train_tgt ../webnlg/webnlg-baseline/train-webnlg-all-delex.lex -valid_src ../webnlg/webnlg-baseline/dev-webnlg-all-delex.triple -valid_tgt ../webnlg/webnlg-baseline/dev-webnlg-all-delex.lex -src_seq_length 70 -tgt_seq_length 70 -save_data baseline

th preprocess.lua -train_src ../data_utils/data_process_pack/train-1-webnlg-all-delex.triple -train_tgt ../data_utils/data_process_pack/train-1-webnlg-all-delex.lex -valid_src ../data_utils/data_process_pack/train-2-webnlg-all-delex.triple -valid_tgt ../data_utils/data_process_pack/train-2-webnlg-all-delex.lex -src_seq_length 70 -tgt_seq_length 70 -save_data baseline

th train.lua -data baseline-train.t7 -save_model baseline
cd /mnt/c/Users/zhy-win/Git/OpenNMT
# th translate.lua -model baseline_epoch13_*.t7 -src ../webnlg/src/baseline-official/train-webnlg-all-delex.triple -output baseline_predictions.txt
th translate.lua -model baseline_epoch13_*.t7 -src ../data_utils/data_process_pack/dev-webnlg-all-delex.triple -output baseline_predictions_dev.txt
cd /mnt/c/Users/zhy-win/Git/webnlg/src/baseline-official
python webnlg_relexicalise.py -i /mnt/c/Users/zhy-win/Git/webnlg/data/ -f /mnt/c/Users/zhy-win/Git/OpenNMT/baseline_predictions.txt

export TEST_TARGETS_REF0=all-notdelex-reference0.lex
export TEST_TARGETS_REF1=all-notdelex-reference1.lex
export TEST_TARGETS_REF2=all-notdelex-reference2.lex
export TEST_TARGETS_REF3=all-notdelex-reference3.lex
export TEST_TARGETS_REF4=all-notdelex-reference4.lex
export TEST_TARGETS_REF5=all-notdelex-reference5.lex
export TEST_TARGETS_REF6=all-notdelex-reference6.lex
export TEST_TARGETS_REF7=all-notdelex-reference7.lex

./multi-bleu.perl ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2} ${TEST_TARGETS_REF3} ${TEST_TARGETS_REF4} ${TEST_TARGETS_REF5} ${TEST_TARGETS_REF6} ${TEST_TARGETS_REF7} < relexicalised_predictions.txt



cd /mnt/c/Users/zhy-win/Git/webnlg/webnlg-baseline
python webnlg_relexicalise.py -i /mnt/c/Users/zhy-win/Git/webnlg/data/ -f /mnt/c/Users/zhy-win/Git/webnlg/webnlg-baseline/predictions.txt

./multi-bleu.perl ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2} ${TEST_TARGETS_REF3} ${TEST_TARGETS_REF4} ${TEST_TARGETS_REF5} ${TEST_TARGETS_REF6} ${TEST_TARGETS_REF7} < relexicalised_predictions.txt
