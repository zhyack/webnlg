cd mnt/c/Users/zhy-win/Git/OpenNMT
th preprocess.lua -train_src ../webnlg/webnlg-baseline/train-webnlg-all-delex.triple -train_tgt ../webnlg/webnlg-baseline/train-webnlg-all-delex.lex -valid_src ../webnlg/webnlg-baseline/dev-webnlg-all-delex.triple -valid_tgt ../webnlg/webnlg-baseline/dev-webnlg-all-delex.lex -src_seq_length 70 -tgt_seq_length 70 -save_data baseline
th train.lua -data baseline-train.t7 -save_model baseline
th translate.lua -model baseline_epoch13_*.t7 -src ../webnlg/webnlg-baseline/dev-webnlg-all-delex.triple -output baseline_predictions.txt
cd mnt/c/Users/zhy-win/Git/webnlg/webnlg-baseline
python webnlg_relexicalise.py -i mnt/c/Users/zhy-win/Git/webnlg/webnlg-baseline/ -f mnt/c/Users/zhy-win/Git/OpenNMT/baseline_predictions.txt
