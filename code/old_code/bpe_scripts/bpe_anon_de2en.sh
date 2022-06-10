#!/bin/bash

DATA="/data/s3225143/data/1_main_data"
DATA_TRAIN="/data/s3225143/data/3_anonymized/full"
CODES="/data/s3225143/models/wmt19.de-en.joined-dict.ensemble/bpecodes"
DICT="/data/s3225143/models/wmt19.de-en.joined-dict.ensemble"
OUTPATH="/data/s3225143/data/3_anonymized/bpe/de-en"

cd $OUTPATH/EMEA
rm *
cd ../JRC
rm *
cd ../GNOME
rm *

cd /data/s3225143/data/fastBPE

./fast applybpe "$OUTPATH/EMEA/train.bpe.de" "$DATA_TRAIN/EMEA.de-en.de" $CODES
./fast applybpe "$OUTPATH/EMEA/train.bpe.en" "$DATA_TRAIN/EMEA.de-en.en" $CODES
./fast applybpe "$OUTPATH/JRC/train.bpe.de" "$DATA_TRAIN/JRC.de-en.de" $CODES
./fast applybpe "$OUTPATH/JRC/train.bpe.en" "$DATA_TRAIN/JRC.de-en.en" $CODES
./fast applybpe "$OUTPATH/GNOME/train.bpe.de" "$DATA_TRAIN/GNOME.de-en.de" $CODES
./fast applybpe "$OUTPATH/GNOME/train.bpe.en" "$DATA_TRAIN/GNOME.de-en.en" $CODES

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
