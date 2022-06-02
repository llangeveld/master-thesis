#!/bin/bash

DATA="/data/s3225143/data/1_main_data"
CODES="/data/s3225143/models/wmt19.en-de.joined-dict.ensemble/bpecodes"
DICT="/data/s3225143/models/wmt19.en-de.joined-dict.ensemble"
OUTPATH="/data/s3225143/code/shell/out"


cd /data/s3225143/data/fastBPE

# EMEA en-de

./fast applybpe "$OUTPATH/EMEA/train.bpe.de" "$DATA/EMEA/train.de" $CODES
./fast getvocab "$OUTPATH/EMEA/train.bpe.de" > "$OUTPATH/EMEA/dict_de.txt"

./fast applybpe "$OUTPATH/EMEA/test.bpe.de" "$DATA/EMEA/test.de" $CODES "$OUTPATH/EMEA/dict_de.txt"
./fast applybpe "$OUTPATH/EMEA/valid.bpe.de" "$DATA/EMEA/valid.de" $CODES "$OUTPATH/EMEA/dict_de.txt"

./fast applybpe "$OUTPATH/EMEA/train.bpe.en" "$DATA/EMEA/train.en" $CODES
./fast getvocab "$OUTPATH/EMEA/train.bpe.en" > "$OUTPATH/EMEA/dict_en.txt"
./fast applybpe "$OUTPATH/EMEA/test.bpe.en" "$DATA/EMEA/test.en" $CODES "$OUTPATH/EMEA/dict_en.txt"
./fast applybpe "$OUTPATH/EMEA/valid.bpe.en" "$DATA/EMEA/valid.en" $CODES "$OUTPATH/EMEA/dict_en.txt"

fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $OUTPATH/EMEA/train.bpe --validpref $OUTPATH/EMEA/valid.bpe --testpref $OUTPATH/EMEA/test.bpe \
  --destdir $OUTPATH/EMEA.en-de
  
 # EMEA de-en
 
./fast applybpe "$OUTPATH/EMEA/train.bpe.de" "$DATA/EMEA/train.de" $CODES
./fast getvocab "$OUTPATH/EMEA/train.bpe.de" > "$OUTPATH/EMEA/dict_de.txt"

./fast applybpe "$OUTPATH/EMEA/test.bpe.de" "$DATA/EMEA/test.de" $CODES "$OUTPATH/EMEA/dict_de.txt"
./fast applybpe "$OUTPATH/EMEA/valid.bpe.de" "$DATA/EMEA/valid.de" $CODES "$OUTPATH/EMEA/dict_de.txt"

./fast applybpe "$OUTPATH/EMEA/train.bpe.en" "$DATA/EMEA/train.en" $CODES
./fast getvocab "$OUTPATH/EMEA/train.bpe.en" > "$OUTPATH/EMEA/dict_en.txt"
./fast applybpe "$OUTPATH/EMEA/test.bpe.en" "$DATA/EMEA/test.en" $CODES "$OUTPATH/EMEA/dict_en.txt"
./fast applybpe "$OUTPATH/EMEA/valid.bpe.en" "$DATA/EMEA/valid.en" $CODES "$OUTPATH/EMEA/dict_en.txt"

fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $OUTPATH/EMEA/train.bpe --validpref $OUTPATH/EMEA/valid.bpe --testpref $OUTPATH/EMEA/test.bpe \
  --destdir $OUTPATH/EMEA.en-de
