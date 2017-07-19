cd /Users/harold/Personal/Project_NLG/Baseline
python3 webnlg_baseline_input.py -i  /Users/harold/Personal/Project_NLG/Baseline/challenge_data_train_dev/
python3 Baseline_Retrieval.py
python3 webnlg_relexicalise.py -i /Users/harold/Personal/Project_NLG/Baseline/challenge_data_train_dev/ -f /Users/harold/Personal/Project_NLG/Baseline/baseline_predictions.txt

export TEST_TARGETS_REF0=all-notdelex-reference0.lex
export TEST_TARGETS_REF1=all-notdelex-reference1.lex
export TEST_TARGETS_REF2=all-notdelex-reference2.lex
export TEST_TARGETS_REF3=all-notdelex-reference3.lex
export TEST_TARGETS_REF4=all-notdelex-reference4.lex
export TEST_TARGETS_REF5=all-notdelex-reference5.lex
export TEST_TARGETS_REF6=all-notdelex-reference6.lex
export TEST_TARGETS_REF7=all-notdelex-reference7.lex

./multi-bleu.perl ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2} ${TEST_TARGETS_REF3} ${TEST_TARGETS_REF4} ${TEST_TARGETS_REF5} ${TEST_TARGETS_REF6} ${TEST_TARGETS_REF7} < relexicalised_predictions.txt

