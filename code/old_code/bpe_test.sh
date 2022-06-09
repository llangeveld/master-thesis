#!/bin/bash

DATA="/data/s3225143/data/1_main_data/tokenized/EMEA"
CODES="/data/s3225143/models/wmt19.en-de.joined-dict.ensemble/bpecodes"
DICT="/data/s3225143/models/wmt19.en-de.joined-dict.ensemble"
OUTPATH="/data/s3225143/data/bpe_test/EMEA"


cd /data/s3225143/data/fastBPE

# applybpe: outfile - input - codes - dict
# EMEA en-de
./fast learnbpe 20000 $DATA/train.de $DATA/train.en > $OUTPATH/codes

./fast applybpe $OUTPATH/train.bpe.de $OUTPATH/train.de $OUTPATH/codes
./fast applybpe $OUTPATH/train.bpe.en $OUTPATH/train.en $OUTPATH/codes

./fast getvocab $OUTPATH/train.bpe.de > $OUTPATH/dict.de
./fast getvocab $OUTPATH/train.bpe.en > $OUTPATH/dict.en

./fast applybpe $OUTPATH/valid.bpe.de $DATA/valid.de $OUTPATH/codes $OUTPATH/dict.de
./fast applybpe $OUTPATH/valid.bpe.en $DATA/valid.en $OUTPATH/codes $OUTPATH/dict.en
./fast applybpe $OUTPATH/test.bpe.de $DATA/test.de $OUTPATH/codes $OUTPATH/dict.de
./fast applybpe $OUTPATH/test.bpe.en $DATA/test.en $OUTPATH/codes $OUTPATH/dict.en


fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $OUTPATH/EMEA/train.bpe --validpref $OUTPATH/EMEA/valid.bpe --testpref $OUTPATH/EMEA/test.bpe \
  --destdir $OUTPATH/EMEA.en-de
