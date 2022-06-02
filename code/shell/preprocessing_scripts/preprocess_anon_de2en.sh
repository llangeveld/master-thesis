#!/bin/bash
TEXT=/data/s3225143/data/3_anonymized/bpe/de-en
DEST=/data/s3225143/data/3_anonymized/processed

# EMEA
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/EMEA/train.bpe --validpref $TEXT/EMEA/valid.bpe --testpref $TEXT/EMEA/test.bpe \
  --destdir $DEST/EMEA.de-en
 
# JRC
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/JRC/train.bpe --validpref $TEXT/JRC/valid.bpe --testpref $TEXT/JRC/test.bpe \
  --destdir $DEST/JRC.de-en

# GNOME
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/GNOME/train.bpe --validpref $TEXT/GNOME/valid.bpe --testpref $TEXT/GNOME/test.bpe \
  --destdir $DEST/GNOME.de-en
