#!/bin/bash
TEXT=/data/s3225143/data/model_bpe/
DEST=/data/s3225143/data/data_preprocessed

# EMEA
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/EMEA/train.bpe --validpref $TEXT/EMEA/valid.bpe --testpref $TEXT/EMEA/test.bpe \
  --destdir $DEST/EMEA.de-en
  
fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $TEXT/EMEA/train.bpe --validpref $TEXT/EMEA/valid.bpe --testpref $TEXT/EMEA/test.bpe \
  --destdir $DEST/EMEA.en-de

# JRC
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/JRC/train.bpe --validpref $TEXT/JRC/valid.bpe --testpref $TEXT/JRC/test.bpe \
  --destdir $DEST/JRC.de-en
  
fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $TEXT/JRC/train.bpe --validpref $TEXT/JRC/valid.bpe --testpref $TEXT/JRC/test.bpe \
  --destdir $DEST/JRC.en-de

# GNOME
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/GNOME/train.bpe --validpref $TEXT/GNOME/valid.bpe --testpref $TEXT/GNOME/test.bpe \
  --destdir $DEST/GNOME.de-en
  
fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $TEXT/GNOME/train.bpe --validpref $TEXT/GNOME/valid.bpe --testpref $TEXT/GNOME/test.bpe \
  --destdir $DEST/GNOME.en-de
