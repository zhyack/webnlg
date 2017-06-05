export VOCAB_SOURCE=../webnlg/modify/dict_src
export VOCAB_TARGET=../webnlg/modify/dict_dst
export TRAIN_SOURCES=../webnlg/modify/train-webnlg-all-delex.triple
export TRAIN_TARGETS=../webnlg/modify/train-mod.txt
export DEV_SOURCES=../webnlg/modify/dev-webnlg-all-delex.triple
export DEV_TARGETS=../webnlg/modify/dev-mod.txt
export DEV_TARGETS_REF=../webnlg/modify/dev-mod.txt
export TRAIN_STEPS=20000
export MODEL_DIR=./yymodel
export EVAL_N=500
python -m bin.train   --config_paths="
      ./example_configs/nmt_large_3377.yml,
      ./example_configs/train_seq2seq_70.yml,
      ./example_configs/text_metrics_simple.yml"   --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET"   --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS"   --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS"   --batch_size 64   --train_steps $TRAIN_STEPS  --output_dir $MODEL_DIR --eval_every_n_steps $EVAL_N

export VOCAB_SOURCE=../webnlg/modify/dict_src
export VOCAB_TARGET=../webnlg/modify/dict_dst
export TRAIN_SOURCES=../webnlg/modify/train-webnlg-all-delex.triple
export TRAIN_TARGETS=../webnlg/modify/train-mod.txt
export DEV_SOURCES=../webnlg/modify/dev-webnlg-all-delex.triple
export DEV_TARGETS=../webnlg/modify/dev-mod.txt
export DEV_TARGETS_REF=../webnlg/modify/dev-mod.txt
export TRAIN_STEPS=120000
export MODEL_DIR=./zzmodel
export EVAL_N=500
python -m bin.train   --config_paths="
      ./example_configs/nmt_large_70.yml,
      ./example_configs/train_seq2seq_70.yml,
      ./example_configs/text_metrics_simple.yml"   --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET"   --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS"   --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS"   --batch_size 32   --train_steps $TRAIN_STEPS  --output_dir $MODEL_DIR --eval_every_n_steps $EVAL_N

export VOCAB_SOURCE=../webnlg/modify/dict_src
export VOCAB_TARGET=../webnlg/modify/dict_dst
export TRAIN_SOURCES=../webnlg/modify/train-webnlg-all-delex.triple
export TRAIN_TARGETS=../webnlg/modify/train-mod.txt
export DEV_SOURCES=../webnlg/modify/dev-webnlg-all-delex.triple
export DEV_TARGETS=../webnlg/modify/dev-mod.txt
export DEV_TARGETS_REF=../webnlg/modify/dev-mod.txt
export TRAIN_STEPS=120000
export MODEL_DIR=./offmodel
export EVAL_N=2000
python -m bin.train   --config_paths="
      ./example_configs/nmt_large.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_simple.yml"   --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET"   --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS"   --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS"   --batch_size 32   --train_steps $TRAIN_STEPS  --output_dir $MODEL_DIR --eval_every_n_steps $EVAL_N

export PRED_DIR=~/webnlg/modify
mkdir -p ${PRED_DIR}
cd ~/seq2seq/
python -m bin.infer \
  --tasks "
    - class: DecodeText" \
  --model_dir $MODEL_DIR \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $DEV_SOURCES" \
  >  ${PRED_DIR}/predictions.txt

cd ~/webnlg/
python data_utils.py

cd ~/seq2seq/
./bin/tools/multi-bleu.perl ${DEV_TARGETS_REF} < ${PRED_DIR}/predictions.txt
# ./bin/tools/multi-bleu.perl ${TRAIN_TARGETS} < ${PRED_DIR}/predictions.txt

./bin/tools/multi-bleu.perl ${PRED_DIR}/dev-webnlg-all-notdelex.lex < ${PRED_DIR}/predict_full.txt

export TEST_TARGETS_REF0=all-notdelex-reference0.lex
export TEST_TARGETS_REF1=all-notdelex-reference1.lex
export TEST_TARGETS_REF2=all-notdelex-reference2.lex
export TEST_TARGETS_REF3=all-notdelex-reference3.lex
export TEST_TARGETS_REF4=all-notdelex-reference4.lex
export TEST_TARGETS_REF5=all-notdelex-reference5.lex
export TEST_TARGETS_REF6=all-notdelex-reference6.lex
export TEST_TARGETS_REF7=all-notdelex-reference7.lex
cd ~/webnlg/webnlg-baseline/
python webnlg_relexicalise.py -i ../data/ -f ${PRED_DIR}/predictions.txt

~/seq2seq/bin/tools/multi-bleu.perl ${TEST_TARGETS_REF0} ${TEST_TARGETS_REF1} ${TEST_TARGETS_REF2} ${TEST_TARGETS_REF3} ${TEST_TARGETS_REF4} ${TEST_TARGETS_REF5} ${TEST_TARGETS_REF6} ${TEST_TARGETS_REF7} < relexicalised_predictions.txt



export VOCAB_SOURCE=../webnlg/data/dict_src
export VOCAB_TARGET=../webnlg/data/dict_dst
export TRAIN_SOURCES=../webnlg/data/train_input_text.txt
export TRAIN_TARGETS=../webnlg/data/train_output_text.txt
export DEV_SOURCES=../webnlg/data/dev_input_text.txt
export DEV_TARGETS=../webnlg/data/dev_output_text.txt
export DEV_TARGETS_REF=../webnlg/data/dev_output_text.txt
export TRAIN_STEPS=120000
export MODEL_DIR=./largemodel
export EVAL_N=2000
python -m bin.train   --config_paths="
      ./example_configs/nmt_large.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_simple.yml"   --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET"   --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS"   --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS"   --batch_size 32   --train_steps $TRAIN_STEPS  --output_dir $MODEL_DIR --eval_every_n_steps $EVAL_N

export PRED_DIR=../webnlg/data
mkdir -p ${PRED_DIR}
cd ../seq2seq/
python -m bin.infer \
  --tasks "
    - class: DecodeText" \
  --model_dir $MODEL_DIR \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $DEV_SOURCES" \
  >  ${PRED_DIR}/predictions.txt

cd ../webnlg/
python data_utils.py

cd ../seq2seq/
./bin/tools/multi-bleu.perl ${DEV_TARGETS_REF} < ${PRED_DIR}/predictions.txt
# ./bin/tools/multi-bleu.perl ${TRAIN_TARGETS} < ${PRED_DIR}/predictions.txt

./bin/tools/multi-bleu.perl ${PRED_DIR}/dev_output_text_ori.txt < ${PRED_DIR}/predict_full.txt
# ./bin/tools/multi-bleu.perl ${PRED_DIR}/dev_output_text_ori_s.txt < ${PRED_DIR}/predict_full.txt
./bin/tools/multi-bleu.perl ${PRED_DIR}/dev_output_text_ori.txt0 ${PRED_DIR}/dev_output_text_ori.txt1 ${PRED_DIR}/dev_output_text_ori.txt2 ${PRED_DIR}/dev_output_text_ori.txt3 ${PRED_DIR}/dev_output_text_ori.txt4 ${PRED_DIR}/dev_output_text_ori.txt5 ${PRED_DIR}/dev_output_text_ori.txt6 ${PRED_DIR}/dev_output_text_ori.txt7 < ${PRED_DIR}/predict_full.txt


export VOCAB_SOURCE=../webnlg/data/dict_src
export VOCAB_TARGET=../webnlg/data/dict_dst
export TRAIN_SOURCES=../webnlg/data/train_input_text.txt
export TRAIN_TARGETS=../webnlg/data/train_output_text.txt
export DEV_SOURCES=../webnlg/data/dev_input_text.txt
export DEV_TARGETS=../webnlg/data/dev_output_text.txt
export DEV_TARGETS_REF=../webnlg/data/dev_output_text.txt
export TRAIN_STEPS=120000
export MODEL_DIR=./model
export EVAL_N=2000
python -m bin.train   --config_paths="
      ./example_configs/nmt_medium.yml,
      ./example_configs/train_seq2seq.yml,
      ./example_configs/text_metrics_bpe.yml"   --model_params "
      vocab_source: $VOCAB_SOURCE
      vocab_target: $VOCAB_TARGET"   --input_pipeline_train "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $TRAIN_SOURCES
      target_files:
        - $TRAIN_TARGETS"   --input_pipeline_dev "
    class: ParallelTextInputPipeline
    params:
       source_files:
        - $DEV_SOURCES
       target_files:
        - $DEV_TARGETS"   --batch_size 32   --train_steps $TRAIN_STEPS  --output_dir $MODEL_DIR --eval_every_n_steps $EVAL_N

export PRED_DIR=${MODEL_DIR}/pred
mkdir -p ${PRED_DIR}
python -m bin.infer \
  --tasks "
    - class: DecodeText" \
  --model_dir $MODEL_DIR \
  --input_pipeline "
    class: ParallelTextInputPipeline
    params:
      source_files:
        - $DEV_SOURCES" \
  >  ${PRED_DIR}/predictions.txt

./bin/tools/multi-bleu.perl ${DEV_TARGETS_REF} < ${PRED_DIR}/predictions.txt

./bin/tools/multi-bleu.perl ${PRED_DIR}/dev_output_text_ori.txt < ${PRED_DIR}/predict_full.txt
