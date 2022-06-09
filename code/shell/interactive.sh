#!/bin/bash
START=/data/s3225143
DATA=$START/data/4_translation/processed/EMEA.en-de
MODEL_DIR=$START/models/finetune/en-de/EMEA/checkpoint_best.pt

fairseq-generate $DATA \
  --path $MODEL_DIR \
  --source-lang en --target-lang de \
  --tokenizer moses --remove-bpe \
  --sacrebleu > gen.out

grep ^H gen.out | cut -f3- > gen.out.sys
grep ^T gen.out | cut -f2- > gen.out.ref

python3 postprocess_sys.py

sacrebleu gen.out.ref -i gen.out.sys.detok -m bleu -b -w 4g