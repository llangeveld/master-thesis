#!/bin/bash
TEXT=/data/s3225143/data/4_translation/test
DEST=/data/s3225143/data/4_translation/test/processed

# EMEA
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/EMEA/de2en/train --validpref $TEXT/EMEA/de2en/valid --testpref $TEXT/EMEA/de2en/test \
  --destdir $DEST/EMEA.de-en
  
fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $TEXT/EMEA/en2de/train --validpref $TEXT/EMEA/en2de/valid --testpref $TEXT/EMEA/en2de/test \
  --destdir $DEST/EMEA.en-de

# JRC
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/JRC/de2en/train --validpref $TEXT/JRC/de2en/valid --testpref $TEXT/JRC/de2en/test \
  --destdir $DEST/JRC.de-en
  
fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $TEXT/JRC/en2de/train --validpref $TEXT/JRC/en2de/valid --testpref $TEXT/JRC/en2de/test \
  --destdir $DEST/JRC.en-de

# GNOME
fairseq-preprocess --source-lang de --target-lang en \
  --trainpref $TEXT/GNOME/de2en/train --validpref $TEXT/GNOME/de2en/valid --testpref $TEXT/GNOME/de2en/test \
  --destdir $DEST/GNOME.de-en
  
fairseq-preprocess --source-lang en --target-lang de \
  --trainpref $TEXT/GNOME/en2de/train --validpref $TEXT/GNOME/en2de/valid --testpref $TEXT/GNOME/en2de/test \
  --destdir $DEST/GNOME.en-de
