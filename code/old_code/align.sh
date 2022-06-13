DATA=$HOME/thesis/local/data/3_anonymized/fastalign
cd $HOME/thesis/local/data/fast_align/build/

./fast_align -i $DATA/EMEA_test.en-de -d -o -v > $DATA/EMEA_test.en-de.align
./fast_align -i $DATA/EMEA_test.de-en -d -o -v > $DATA/EMEA_test.de-en.align
./fast_align -i $DATA/JRC_test.en-de -d -o -v > $DATA/JRC_test.en-de.align
./fast_align -i $DATA/JRC_test.de-en -d -o -v > $DATA/JRC_test.de-en.align
./fast_align -i $DATA/GNOME_test.en-de -d -o -v > $DATA/GNOME_test.en-de.align
./fast_align -i $DATA/GNOME_test.de-en -d -o -v > $DATA/GNOME_test.de-en.align

./fast_align -i $DATA/EMEA_train.en-de -d -o -v > $DATA/EMEA_train.en-de.align
./fast_align -i $DATA/EMEA_train.de-en -d -o -v > $DATA/EMEA_train.de-en.align
./fast_align -i $DATA/JRC_train.en-de -d -o -v > $DATA/JRC_train.en-de.align
./fast_align -i $DATA/JRC_train.de-en -d -o -v > $DATA/JRC_train.de-en.align
./fast_align -i $DATA/GNOME_train.en-de -d -o -v > $DATA/GNOME_train.en-de.align
./fast_align -i $DATA/GNOME_train.de-en -d -o -v > $DATA/GNOME_train.de-en.align

./fast_align -i $DATA/EMEA_valid.en-de -d -o -v > $DATA/EMEA_valid.en-de.align
./fast_align -i $DATA/EMEA_valid.de-en -d -o -v > $DATA/EMEA_valid.de-en.align
./fast_align -i $DATA/JRC_valid.en-de -d -o -v > $DATA/JRC_valid.en-de.align
./fast_align -i $DATA/JRC_valid.de-en -d -o -v > $DATA/JRC_valid.de-en.align
./fast_align -i $DATA/GNOME_valid.en-de -d -o -v > $DATA/GNOME_valid.en-de.align
./fast_align -i $DATA/GNOME_valid.de-en -d -o -v > $DATA/GNOME_valid.de-en.align