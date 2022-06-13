#!/bin/bash
START=/data/s3225143
DOMAIN="EMEA"
DATA=$START/data/4_translation/single-processed/$DOMAIN.en-de
MODEL_DIR=$START/models/single-en-de/model.pt
WRITE=$START/code/shell/finetune/single-en/de

fairseq-generate $DATA \
  --path $MODEL_DIR \
  --source-lang en --target-lang de \
  --tokenizer moses --remove-bpe \
  --sacrebleu > $WRITE/gen_$DOMAIN.out

grep ^H $WRITE/gen_$DOMAIN.out | cut -f3- > $WRITE/gen_$DOMAIN.out.sys
grep ^T $WRITE/gen_$DOMAIN.out | cut -f2- > $WRITE/gen_$DOMAIN.out.ref

python3 postprocess_sys.py -i $WRITE/gen_$DOMAIN.out.sys -o $WRITE/gen_$DOMAIN.out.sys.detok

sacrebleu $WRITE/gen_$DOMAIN.out.ref -i $WRITE/gen_$DOMAIN.out.sys.detok -m bleu -b -w 4

DOMAIN="JRC"
DATA=$START/data/4_translation/single-processed/$DOMAIN.en-de

fairseq-generate $DATA \
  --path $MODEL_DIR \
  --source-lang en --target-lang de \
  --tokenizer moses --remove-bpe \
  --sacrebleu > $WRITE/gen_$DOMAIN.out

grep ^H $WRITE/gen_$DOMAIN.out | cut -f3- > $WRITE/gen_$DOMAIN.out.sys
grep ^T $WRITE/gen_$DOMAIN.out | cut -f2- > $WRITE/gen_$DOMAIN.out.ref

python3 postprocess_sys.py -i $WRITE/gen_$DOMAIN.out.sys -o $WRITE/gen_$DOMAIN.out.sys.detok

sacrebleu $WRITE/gen_$DOMAIN.out.ref -i $WRITE/gen_$DOMAIN.out.sys.detok -m bleu -b -w 4

DOMAIN="GNOME"
DATA=$START/data/4_translation/single-processed/$DOMAIN.en-de

fairseq-generate $DATA \
  --path $MODEL_DIR \
  --source-lang en --target-lang de \
  --tokenizer moses --remove-bpe \
  --sacrebleu > $WRITE/gen_$DOMAIN.out

grep ^H $WRITE/gen_$DOMAIN.out | cut -f3- > $WRITE/gen_$DOMAIN.out.sys
grep ^T $WRITE/gen_$DOMAIN.out | cut -f2- > $WRITE/gen_$DOMAIN.out.ref

python3 postprocess_sys.py -i $WRITE/gen_$DOMAIN.out.sys -o $WRITE/gen_$DOMAIN.out.sys.detok

sacrebleu $WRITE/gen_$DOMAIN.out.ref -i $WRITE/gen_$DOMAIN.out.sys.detok -m bleu -b -w 4
