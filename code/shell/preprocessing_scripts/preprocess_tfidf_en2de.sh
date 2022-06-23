#!/bin/bash

DATA="/data/s3225143/data/1_main_data/tokenized"
DATA_TRAIN="/data/s3225143/data/3_anonymized/tfidf"
CODES="/data/s3225143/models/single-en-de/bpecodes"
DICT="/data/s3225143/models/single-en-de/"
OUTPATH="/data/s3225143/data/4_translation/bpe-tfidf/en-de"
DEST=/data/s3225143/data/4_translation/processed-tfidf

cd /data/s3225143/data/fastBPE

./fast applybpe "$OUTPATH/EMEA/train.bpe.de" "$DATA_TRAIN/EMEA.en-de.de" $CODES "$DICT/dict.de.txt"
./fast applybpe "$OUTPATH/EMEA/train.bpe.en" "$DATA_TRAIN/EMEA.en-de.en" $CODES "$DICT/dict.en.txt"
./fast applybpe "$OUTPATH/JRC/train.bpe.de" "$DATA_TRAIN/JRC.en-de.de" $CODES "$DICT/dict.de.txt"
./fast applybpe "$OUTPATH/JRC/train.bpe.en" "$DATA_TRAIN/JRC.en-de.en" $CODES "$DICT/dict.en.txt"
./fast applybpe "$OUTPATH/GNOME/train.bpe.de" "$DATA_TRAIN/GNOME.en-de.de" $CODES "$DICT/dict.en.txt"
./fast applybpe "$OUTPATH/GNOME/train.bpe.en" "$DATA_TRAIN/GNOME.en-de.en" $CODES "$DICT/dict.en.txt"

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
fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $OUTPATH/EMEA/train.bpe --validpref $OUTPATH/EMEA/valid.bpe --testpref $OUTPATH/EMEA/test.bpe \
  --destdir $DEST/EMEA.en-de --tgtdict $DICT/dict.de.txt --srcdict $DICT/dict.en.txt \
  --tokenizer moses --bpe fastbpe

# JRC
fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $OUTPATH/JRC/train.bpe --validpref $OUTPATH/JRC/valid.bpe --testpref $OUTPATH/JRC/test.bpe \
  --destdir $DEST/JRC.en-de --tgtdict $DICT/dict.de.txt --srcdict $DICT/dict.en.txt \
  --tokenizer moses --bpe fastbpe

# GNOME
fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $OUTPATH/GNOME/train.bpe --validpref $OUTPATH/GNOME/valid.bpe --testpref $OUTPATH/GNOME/test.bpe \
  --destdir $DEST/GNOME.en-de --tgtdict $DICT/dict.de.txt --srcdict $DICT/dict.en.txt \
  --tokenizer moses --bpe fastbpe
