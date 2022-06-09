#!/bin/bash
START=/data/s3225143/data/bpe_test/EMEA
DATA=$START/EMEA.en-de
MODEL_DIR=$START/models-large/checkpoint_best.pt

# Generate translations
fairseq-generate $DATA \
  --path $MODEL_DIR \
  --source-lang en --target-lang de \
  --tokenizer moses --remove-bpe \
  --sacrebleu > gen.out

# Postprocess out-file
grep ^H gen.out | cut -f3- > gen.out.sys
grep ^T gen.out | cut -f2- > gen.out.ref

# De-tokenize hypotheses
python3 postprocess_sys.py

# Score
sacrebleu gen.out.ref -i gen.out.sys.detok -m bleu -b -w 4g
