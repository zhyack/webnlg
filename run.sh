export VOCAB_SOURCE=../webnlg/data/dict_src
export VOCAB_TARGET=../webnlg/data/dict_dst
export TRAIN_SOURCES=../webnlg/data/train_input_text.txt
export TRAIN_TARGETS=../webnlg/data/train_output_text.txt
export DEV_SOURCES=../webnlg/data/dev_input_text.txt
export DEV_TARGETS=../webnlg/data/dev_output_text.txt
export DEV_TARGETS_REF=../webnlg/data/dev_output_text.txt
export TRAIN_STEPS=100000
export MODEL_DIR=./model
export EVAL_N=1000
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
