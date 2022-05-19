#!/bin/bash
START=/data/s3225143
DATA_EMEA=$START/data/4_translation/processed/EMEA.en-de/
DATA_JRC=$START/data/4_translation/processed/JRC.en-de/
DATA_GNOME=$START/data/4_translation/processed/GNOME.en-de/
MODELS=$START/models
PRETRAINED=$MODELS/wmt19.en-de.joined-dict.ensemble/model1.pt

CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA_EMEA \
  --seed 2 --fp16 --left-pad-source \
  --no-progress-bar --log-format simple --log-interval 100 \
  --finetune-from-model $PRETRAINED --task translation \
  --arch transformer_wmt_en_de_big --source-lang en --target-lang de \
  --save-dir /data/s3225143/models/finetune/EMEA \
  --ddp-backend c10d --max-tokens 3584 --max-tokens-valid 3584 --update-epoch-batch-itr True \
  --max-update 202200 --update-freq 1 --save-interval-updates 200 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --ignore-prefix-size 0 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --relu-dropout 0.0 --dropout 0.2 \
  --weight-decay 0.0 --attention-dropout 0.01 \
  --lr-scheduler inverse_sqrt --lr 0.0007 --warmup-updates 4000 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 \
  --bpe fastbpe --bpe-codes /data/s3225143/models/wmt19.en-de.joined-dict.ensemble/bpecodes \
  --tokenizer moses \
  --decoder-attention-heads 16 --decoder-embed-dim 1024 --decoder-ffn-embed-dim 4096 \
  --decoder-layers 6 --decoder-output-dim 1024 \
  --encoder-attention-heads 16 --encoder-embed-dim 1024 --encoder-ffn-embed-dim 8192 --encoder-layers 6 \
  --max-source-positions 1024 --max-target-positions 1024 \
  --share-decoder-input-output-embed \
  --upsample-primary 1

