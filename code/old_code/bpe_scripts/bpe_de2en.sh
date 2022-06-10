#!/bin/bash

DATA="/data/s3225143/data/1_main_data"
CODES="/data/s3225143/models/wmt19.de-en.joined-dict.ensemble/bpecodes"
DICT="/data/s3225143/models/wmt19.de-en.joined-dict.ensemble"
OUTPATH="/data/s3225143/data/4_translation/bpe/de-en"

cd $OUTPATH/EMEA
rm *
cd ../JRC
rm *
cd ../GNOME
rm *

cd /data/s3225143/data/fastBPE

./fast applybpe "$OUTPATH/EMEA/train.bpe.de" "$DATA/EMEA/train.de" $CODES
./fast applybpe "$OUTPATH/EMEA/train.bpe.en" "$DATA/EMEA/train.en" $CODES
./fast applybpe "$OUTPATH/JRC/train.bpe.de" "$DATA/JRC/train.de" $CODES
./fast applybpe "$OUTPATH/JRC/train.bpe.en" "$DATA/JRC/train.en" $CODES
./fast applybpe "$OUTPATH/GNOME/train.bpe.de" "$DATA/GNOME/train.de" $CODES
./fast applybpe "$OUTPATH/GNOME/train.bpe.en" "$DATA/GNOME/train.en" $CODES

./fast getvocab "$OUTPATH/EMEA/train.bpe.de" > "$OUTPATH/EMEA/dict_de.txt"
./fast getvocab "$OUTPATH/EMEA/train.bpe.en" > "$OUTPATH/EMEA/dict_en.txt"

./fast applybpe "$OUTPATH/EMEA/valid.bpe.de" "$DATA/EMEA/valid.de" $CODES "$OUTPATH/EMEA/dict_de.txt"
./fast applybpe "$OUTPATH/EMEA/valid.bpe.en" "$DATA/EMEA/valid.en" $CODES "$OUTPATH/EMEA/dict_en.txt"
./fast applybpe "$OUTPATH/JRC/valid.bpe.de" "$DATA/JRC/valid.de" $CODES "$OUTPATH/EMEA/dict_de.txt"
./fast applybpe "$OUTPATH/JRC/valid.bpe.en" "$DATA/JRC/valid.en" $CODES "$OUTPATH/EMEA/dict_en.txt"
./fast applybpe "$OUTPATH/GNOME/valid.bpe.de" "$DATA/GNOME/valid.de" $CODES "$OUTPATH/EMEA/dict_de.txt"
./fast applybpe "$OUTPATH/GNOME/valid.bpe.en" "$DATA/GNOME/valid.en" $CODES "$OUTPATH/EMEA/dict_en.txt"

./fast applybpe "$OUTPATH/EMEA/test.bpe.de" "$DATA/EMEA/test.de" $CODES "$OUTPATH/EMEA/dict_de.txt"
./fast applybpe "$OUTPATH/EMEA/test.bpe.en" "$DATA/EMEA/test.en" $CODES "$OUTPATH/EMEA/dict_en.txt"
./fast applybpe "$OUTPATH/JRC/test.bpe.de" "$DATA/JRC/test.de" $CODES "$OUTPATH/EMEA/dict_de.txt"
./fast applybpe "$OUTPATH/JRC/test.bpe.en" "$DATA/JRC/test.en" $CODES "$OUTPATH/EMEA/dict_en.txt"
./fast applybpe "$OUTPATH/GNOME/test.bpe.de" "$DATA/GNOME/test.de" $CODES "$OUTPATH/EMEA/dict_de.txt"
./fast applybpe "$OUTPATH/GNOME/test.bpe.en" "$DATA/GNOME/test.en" $CODES "$OUTPATH/EMEA/dict_en.txt"
