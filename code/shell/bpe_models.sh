#!/bin/bash

DATA="/data/s3225143/data/1_main_data"
CODES="/data/s3225143/models/wmt19.en-de.joined-dict.ensemble/bpecodes"
DICT="/data/s3225143/models/wmt19.en-de.joined-dict.ensemble"
OUTPATH="/data/s3225143/data/4_translation/test"

cd $OUTPATH/EMEA
rm *
cd ../JRC
rm *
cd ../GNOME
rm *

cd /data/s3225143/data/fastBPE

./fast applybpe "$OUTPATH/EMEA/train.bpe.de" "$DATA/EMEA/train.de" $CODES "$DICT/dict.de.txt"
./fast applybpe "$OUTPATH/EMEA/train.bpe.en" "$DATA/EMEA/train.en" $CODES "$DICT/dict.en.txt"
./fast applybpe "$OUTPATH/JRC/train.bpe.de" "$DATA/JRC/train.de" $CODES "$DICT/dict.de.txt"
./fast applybpe "$OUTPATH/JRC/train.bpe.en" "$DATA/JRC/train.en" $CODES "$DICT/dict.en.txt"
./fast applybpe "$OUTPATH/GNOME/train.bpe.de" "$DATA/GNOME/train.de" $CODES "$DICT/dict.de.txt"
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
