#!/bin/bash
START=/data/s3225143
DOMAIN="EMEA"
DATA=$START/data/3_anonymized/processed/$DOMAIN.en-de
MODEL_DIR=$START/models/finetune-anon/en-de/$DOMAIN/checkpoint_best.pt

fairseq-generate $DATA \
  --path $MODEL_DIR \
  --source-lang en --target-lang de \
  --tokenizer moses --remove-bpe \
  --sacrebleu > gen_$DOMAIN.out

grep ^H gen_$DOMAIN.out | cut -f3- > gen_$DOMAIN.out.sys
grep ^T gen_$DOMAIN.out | cut -f2- > gen_$DOMAIN.out.ref

python3 ../postprocess_sys.py --i gen_$DOMAIN.out.sys --o gen_$DOMAIN.out.sys.detok

sacrebleu gen_$DOMAIN.out.ref -i gen_$DOMAIN.out.sys.detok -m bleu -b -w 4

DOMAIN="JRC"
DATA=$START/data/3_anonymized/processed/$DOMAIN.en-de
MODEL_DIR=$START/models/finetune-anon/en-de/$DOMAIN/checkpoint_best.pt

fairseq-generate $DATA \
  --path $MODEL_DIR \
  --source-lang en --target-lang de \
  --tokenizer moses --remove-bpe \
  --sacrebleu > gen_$DOMAIN.out

grep ^H gen_$DOMAIN.out | cut -f3- > gen_$DOMAIN.out.sys
grep ^T gen_$DOMAIN.out | cut -f2- > gen_$DOMAIN.out.ref

python3 ../postprocess_sys.py --i gen_$DOMAIN.out.sys --o gen_$DOMAIN.out.sys.detok

sacrebleu gen_$DOMAIN.out.ref -i gen_$DOMAIN.out.sys.detok -m bleu -b -w 4

DOMAIN="GNOME"
DATA=$START/data/3_anonymized/processed/$DOMAIN.en-de
MODEL_DIR=$START/models/finetune-anon/en-de/$DOMAIN/checkpoint_best.pt

fairseq-generate $DATA \
  --path $MODEL_DIR \
  --source-lang en --target-lang de \
  --tokenizer moses --remove-bpe \
  --sacrebleu > gen_$DOMAIN.out

grep ^H gen_$DOMAIN.out | cut -f3- > gen_$DOMAIN.out.sys
grep ^T gen_$DOMAIN.out | cut -f2- > gen_$DOMAIN.out.ref

python3 ../postprocess_sys.py --i gen_$DOMAIN.out.sys --o gen_$DOMAIN.out.sys.detok

sacrebleu gen_$DOMAIN.out.ref -i gen_$DOMAIN.out.sys.detok -m bleu -b -w 4