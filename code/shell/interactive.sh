#!/bin/bash
START=/data/s3225143
DATA=$START/data/4_translation/processed/EMEA.en-de
MODEL_DIR=$START/models/finetune/en-de/EMEA/checkpoint_best.pt

CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA \
  --path $MODEL_DIR \
  --source-lang en --target-lang de \
  --tokenizer moses --remove-bpe \
  --sacrebleu | grep -P "D-[0-9]+" > translations.out
