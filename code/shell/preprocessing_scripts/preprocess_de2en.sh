#!/bin/bash

DATA="/data/s3225143/data/1_main_data/tokenized"
CODES="/data/s3225143/models/single-de-en/bpecodes"
DICT="/data/s3225143/models/single-de-en/"
OUTPATH="/data/s3225143/data/4_translation/single-bpe/de-en"
DEST=/data/s3225143/data/4_translation/single-processed

cd /data/s3225143/data/fastBPE

./fast applybpe "$OUTPATH/EMEA/train.bpe.de" "$DATA/EMEA/train.de" $CODES "$DICT/dict.de.txt"
./fast applybpe "$OUTPATH/EMEA/train.bpe.en" "$DATA/EMEA/train.en" $CODES "$DICT/dict.en.txt"
./fast applybpe "$OUTPATH/JRC/train.bpe.de" "$DATA/JRC/train.de" $CODES "$DICT/dict.de.txt"
./fast applybpe "$OUTPATH/JRC/train.bpe.en" "$DATA/JRC/train.en" $CODES "$DICT/dict.en.txt"
./fast applybpe "$OUTPATH/GNOME/train.bpe.de" "$DATA/GNOME/train.de" $CODES "$DICT/dict.en.txt"
./fast applybpe "$OUTPATH/GNOME/train.bpe.en" "$DATA/GNOME/train.en" $CODES "$DICT/dict.en.txt"

./fast applybpe "$OUTPATH/EMEA/valid.bpe.de" "$DATA/EMEA/valid.de" $CODES "$DICT/dict.de.txt"
./fast applybpe "$OUTPATH/EMEA/valid.bpe.en" "$DATA/EMEA/valid.en" $CODES "$DICT/dict.en.txt"
./fast applybpe "$OUTPATH/JRC/valid.bpe.de" "$DATA/JRC/valid.de" $CODES "$DICT/dict.de.txt"
./fast applybpe "$OUTPATH/JRC/valid.bpe.en" "$DATA/JRC/valid.en" $CODES "$DICT/dict.en.txt"
./fast applybpe "$OUTPATH/GNOME/valid.bpe.de" "$DATA/GNOME/valid.de" $CODES "$DICT/dict.de.txt"
./fast applybpe "$OUTPATH/GNOME/valid.bpe.en" "$DATA/GNOME/valid.en" $CODES "$DICT/dict.en.txt"

./fast applybpe "$OUTPATH/EMEA/test.bpe.de" "$DATA/EMEA/test.de" $CODES "$DICT/dict.de.txt"
./fast applybpe "$OUTPATH/EMEA/test.bpe.en" "$DATA/EMEA/test.en" $CODES "$DICT/dict.en.txt"
./fast applybpe "$OUTPATH/JRC/test.bpe.de" "$DATA/JRC/test.de" $CODES "$DICT/dict.de.txt"
./fast applybpe "$OUTPATH/JRC/test.bpe.en" "$DATA/JRC/test.en" $CODES "$DICT/dict.en.txt"
./fast applybpe "$OUTPATH/GNOME/test.bpe.de" "$DATA/GNOME/test.de" $CODES "$DICT/dict.de.txt"
./fast applybpe "$OUTPATH/GNOME/test.bpe.en" "$DATA/GNOME/test.en" $CODES "$DICT/dict.en.txt"

# EMEA
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $OUTPATH/EMEA/train.bpe --validpref $OUTPATH/EMEA/valid.bpe --testpref $OUTPATH/EMEA/test.bpe \
  --destdir $DEST/EMEA.de-en --tgtdict $DICT/dict.de.txt --srcdict $DICT/dict.en.txt \
  --tokenizer moses --bpe fastbpe

# JRC
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $OUTPATH/JRC/train.bpe --validpref $OUTPATH/JRC/valid.bpe --testpref $OUTPATH/JRC/test.bpe \
  --destdir $DEST/JRC.de-en --tgtdict $DICT/dict.de.txt --srcdict $DICT/dict.en.txt \
  --tokenizer moses --bpe fastbpe

# GNOME
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $OUTPATH/GNOME/train.bpe --validpref $OUTPATH/GNOME/valid.bpe --testpref $OUTPATH/GNOME/test.bpe \
  --destdir $DEST/GNOME.de-en --tgtdict $DICT/dict.de.txt --srcdict $DICT/dict.en.txt \
  --tokenizer moses --bpe fastbpe
