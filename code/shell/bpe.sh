#!/bin/bash

DATA="$HOME/thesis/local/data/corrected_data/"
CODES="$HOME/thesis/local/data/bpe_codes/"

cd $HOME/thesis/local/data/bpe_codes/EMEA
rm *
cd ../JRC
rm *
cd ../GNOME
rm *

cd $HOME/thesis/local/data/fastBPE

# Learning BPE for training data (both languages)
./fast learnbpe 2000 "$DATA/EMEA/train.de" "$DATA/EMEA/train.en" > "$CODES/EMEA/codes"
./fast learnbpe 2000 "$DATA/JRC/train.de" "$DATA/JRC/train.en" > "$CODES/JRC/codes"
./fast learnbpe 2000 "$DATA/GNOME/train.de" "$DATA/GNOME/train.en" > "$CODES/GNOME/codes"

# Applying BPE to training data
./fast applybpe "$CODES/EMEA/train.2000.de" "$DATA/EMEA/train.de" "$CODES/EMEA/codes"
./fast applybpe "$CODES/EMEA/train.2000.en" "$DATA/EMEA/train.en" "$CODES/EMEA/codes"
./fast applybpe "$CODES/JRC/train.2000.de" "$DATA/JRC/train.de" "$CODES/JRC/codes"
./fast applybpe "$CODES/JRC/train.2000.en" "$DATA/JRC/train.en" "$CODES/JRC/codes"
./fast applybpe "$CODES/GNOME/train.2000.de" "$DATA/GNOME/train.de" "$CODES/GNOME/codes"
./fast applybpe "$CODES/GNOME/train.2000.en" "$DATA/GNOME/train.en" "$CODES/GNOME/codes"

# Build training vocabulary
./fast getvocab "$CODES/EMEA/train.2000.de" > "$CODES/EMEA/vocab.2000.de"
./fast getvocab "$CODES/EMEA/train.2000.en" > "$CODES/EMEA/vocab.2000.en"
./fast getvocab "$CODES/JRC/train.2000.de" > "$CODES/JRC/vocab.2000.de"
./fast getvocab "$CODES/JRC/train.2000.en" > "$CODES/JRC/vocab.2000.en"
./fast getvocab "$CODES/GNOME/train.2000.de" > "$CODES/GNOME/vocab.2000.de"
./fast getvocab "$CODES/GNOME/train.2000.en" > "$CODES/GNOME/vocab.2000.en"

# Apply BPE to valid and test
./fast applybpe "$CODES/EMEA/valid.2000.de" "$DATA/EMEA/valid.de" "$CODES/EMEA/codes" "$CODES/EMEA/vocab.2000.de"
./fast applybpe "$CODES/EMEA/valid.2000.en" "$DATA/EMEA/valid.en" "$CODES/EMEA/codes" "$CODES/EMEA/vocab.2000.en"
./fast applybpe "$CODES/JRC/valid.2000.de" "$DATA/JRC/valid.de" "$CODES/JRC/codes" "$CODES/JRC/vocab.2000.de"
./fast applybpe "$CODES/JRC/valid.2000.en" "$DATA/JRC/valid.en" "$CODES/JRC/codes" "$CODES/JRC/vocab.2000.en"
./fast applybpe "$CODES/GNOME/valid.2000.de" "$DATA/GNOME/valid.de" "$CODES/GNOME/codes" "$CODES/GNOME/vocab.2000.de"
./fast applybpe "$CODES/GNOME/valid.2000.en" "$DATA/GNOME/valid.en" "$CODES/GNOME/codes" "$CODES/GNOME/vocab.2000.en"

./fast applybpe "$CODES/EMEA/test.2000.de" "$DATA/EMEA/test.de" "$CODES/EMEA/codes" "$CODES/EMEA/vocab.2000.de"
./fast applybpe "$CODES/EMEA/test.2000.en" "$DATA/EMEA/test.en" "$CODES/EMEA/codes" "$CODES/EMEA/vocab.2000.en"
./fast applybpe "$CODES/JRC/test.2000.de" "$DATA/JRC/test.de" "$CODES/JRC/codes" "$CODES/JRC/vocab.2000.de"
./fast applybpe "$CODES/JRC/test.2000.en" "$DATA/JRC/test.en" "$CODES/JRC/codes" "$CODES/JRC/vocab.2000.en"
./fast applybpe "$CODES/GNOME/test.2000.de" "$DATA/GNOME/test.de" "$CODES/GNOME/codes" "$CODES/GNOME/vocab.2000.de"
./fast applybpe "$CODES/GNOME/test.2000.en" "$DATA/GNOME/test.en" "$CODES/GNOME/codes" "$CODES/GNOME/vocab.2000.en"
