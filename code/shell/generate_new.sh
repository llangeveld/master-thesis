#!/bin/bash

DATA=''
MODEL_DIR=''
WRITE=''
DOMAIN=''

while getopts 'i:m:o:d:' flag; do
  case $flag in
    i) DATA=$OPTARG;;
    m) MODEL_DIR=$OPTARG;;
    o) WRITE=${OPTARG};;
    d) DOMAIN=${OPTARG};;
    *) break;;
  esac
done

CUDA_VISIBLE_DEVICES=0 fairseq-generate $DATA \
  --path $MODEL_DIR \
  --source-lang de --target-lang en  \
  --tokenizer moses --remove-bpe \
  --sacrebleu > $WRITE/gen_$DOMAIN.out

grep ^H $WRITE/gen_$DOMAIN.out | cut -f3- > $WRITE/gen_$DOMAIN.out.sys
grep ^T $WRITE/gen_$DOMAIN.out | cut -f2- > $WRITE/gen_$DOMAIN.out.ref

python3 postprocess_sys.py -i $WRITE/gen_$DOMAIN.out.sys -o $WRITE/gen_$DOMAIN.out.sys.detok

sacrebleu $WRITE/gen_$DOMAIN.out.ref -i $WRITE/gen_$DOMAIN.out.sys.detok -m bleu -b -w 4