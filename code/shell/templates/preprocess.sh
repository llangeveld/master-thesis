#!/bin/bash
# Can be run in interactive session!

DATA="/data/s3225143/data/1_main_data/tokenized/"
CODES="/data/s3225143/models/wmt19.en-de.joined-dict.ensemble/bpecodes"
DICT="/data/s3225143/models/wmt19.en-de.joined-dict.ensemble"
OUTPATH="/data/s3225143/data/bpe_test/"
TEXT=/data/s3225143/data/bpe_test/EMEA
DEST=/data/s3225143/data/bpe_test/EMEA

cd /data/s3225143/data/fastBPE

./fast applybpe "$OUTPATH/EMEA/train.bpe.de" "$DATA/EMEA/train.de" $CODES $DICT/dict.de.txt
./fast applybpe "$OUTPATH/EMEA/train.bpe.en" "$DATA/EMEA/train.en" $CODES $DICT/dict.en.txt

./fast applybpe "$OUTPATH/EMEA/valid.bpe.de" "$DATA/EMEA/valid.de" $CODES "$DICT/dict.de.txt"
./fast applybpe "$OUTPATH/EMEA/valid.bpe.en" "$DATA/EMEA/valid.en" $CODES "$DICT/dict.en.txt"

./fast applybpe "$OUTPATH/EMEA/test.bpe.de" "$DATA/EMEA/test.de" $CODES "$DICT/dict.de.txt"
./fast applybpe "$OUTPATH/EMEA/test.bpe.en" "$DATA/EMEA/test.en" $CODES "$DICT/dict.en.txt"

# EMEA
fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $TEXT/train.bpe --validpref $TEXT/valid.bpe --testpref $TEXT/test.bpe \
  --destdir $DEST/EMEA.en-de --tgtdict $DICT/dict.de.txt --srcdict $DICT/dict.en.txt \
  --tokenizer moses --bpe fastbpe
