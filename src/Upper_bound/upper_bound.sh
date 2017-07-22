#!/bin/bash

cd /Users/harold/Personal/Project_NLG/Baseline
python ./upper_bound_calculate_before.py
for k in $( seq 0 870 )
do
	echo ""
	echo "${k}th line"
	echo ""
	python upper_bound_calculate.py
	export TEST_TARGETS_REF0=ref_for_bleu_0.txt
	export TEST_TARGETS_REF1=ref_for_bleu_1.txt
	export TEST_TARGETS_REF2=ref_for_bleu_2.txt
	export TEST_TARGETS_REF3=ref_for_bleu_3.txt
	export TEST_TARGETS_REF4=ref_for_bleu_4.txt
	export TEST_TARGETS_REF5=ref_for_bleu_5.txt
	export TEST_TARGETS_REF6=ref_for_bleu_6.txt
	export TEST_TARGETS_REF7=ref_for_bleu_7.txt
	./multi-bleu-altered.perl ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} < source_for_bleu_0.txt
	./multi-bleu-altered.perl ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} < source_for_bleu_1.txt
	python upper_bound_calculate_after.py
done

export TEST_TARGETS_REF0=all-notdelex-reference0.lex
export TEST_TARGETS_REF1=all-notdelex-reference1.lex
export TEST_TARGETS_REF2=all-notdelex-reference2.lex
export TEST_TARGETS_REF3=all-notdelex-reference3.lex
export TEST_TARGETS_REF4=all-notdelex-reference4.lex
export TEST_TARGETS_REF5=all-notdelex-reference5.lex
export TEST_TARGETS_REF6=all-notdelex-reference6.lex
export TEST_TARGETS_REF7=all-notdelex-reference7.lex

echo "Final Score"
./multi-bleu.perl ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2} ${TEST_TARGETS_REF3} ${TEST_TARGETS_REF4} ${TEST_TARGETS_REF5} ${TEST_TARGETS_REF6} ${TEST_TARGETS_REF7} < output_upper.txt

