#!/bin/bash
START=/data/s3225143
DATA=$START/data/1_main_data/EMEA/test.en
DATA_TRG=$START/data/1_main_data/EMEA/test.de
MODEL_DIR=$START/models/finetune/en-de/EMEA/

fairseq-interactive \
  --path $MODEL_DIR/checkpoint_best.pt $MODEL_DIR \
  --source-lang en --target-lang de \
  --tokenizer moses \
  --input "$DATA" \
  --beam-size 1
