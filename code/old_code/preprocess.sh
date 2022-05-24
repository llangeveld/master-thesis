#!/bin/bash
TEXT=$HOME/thesis/local/data/bpe_codes/
DEST=$HOME/thesis/local/data/fairseq

# EMEA
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/EMEA/train.2000 --validpref $TEXT/EMEA/valid.2000 --testpref $TEXT/EMEA/test.2000 \
  --destdir $DEST/EMEA.de-en
  
fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $TEXT/EMEA/train.2000 --validpref $TEXT/EMEA/valid.2000 --testpref $TEXT/EMEA/test.2000 \
  --destdir $DEST/EMEA.en-de

# JRC
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/JRC/train.2000 --validpref $TEXT/JRC/valid.2000 --testpref $TEXT/JRC/test.2000 \
  --destdir $DEST/JRC.de-en
  
fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $TEXT/JRC/train.2000 --validpref $TEXT/JRC/valid.2000 --testpref $TEXT/JRC/test.2000 \
  --destdir $DEST/JRC.en-de

# GNOME
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/GNOME/train.2000 --validpref $TEXT/GNOME/valid.2000 --testpref $TEXT/GNOME/test.2000 \
  --destdir $DEST/GNOME.de-en
  
fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $TEXT/GNOME/train.2000 --validpref $TEXT/GNOME/valid.2000 --testpref $TEXT/GNOME/test.2000 \
  --destdir $DEST/GNOME.en-de