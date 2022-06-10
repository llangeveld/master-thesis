#!/bin/bash
START=/data/s3225143
DATA=$START/data/3_anonymized/processed/EMEA.en-de/
MODELS=$START/models
THISMODEL=$MODELS/wmt19.en-de.joined-dict.ensemble
PRETRAINED=$THISMODEL/model1.pt
BPECODES=$THISMODEL/bpecodes
SAVE=$MODELS/finetune-anon/en-de/EMEA/

CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA \
  --keep-best-checkpoints 1 --save-interval 100  \
  --keep-interval-updates 1  --keep-last-epochs 1 \
  --keep-best-checkpoints 1   --save-interval-updates 500 \
  --no-epoch-checkpoints \
  --no-save-optimizer-state \
  --no-progress-bar --log-format json --log-interval 100 --log-file $SAVE/log.out\
  --finetune-from-model $PRETRAINED --task translation \
  --arch transformer_wmt_en_de_big --source-lang en --target-lang de \
  --save-dir $SAVE \
  --eval-bleu \
  --eval-bleu-detok moses \
  --eval-bleu-remove-bpe \
  --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
  --ddp-backend c10d --max-tokens 4096 --max-tokens-valid 4096 --update-epoch-batch-itr True \
  --max-epoch 18 --update-freq 1 \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --ignore-prefix-size 0 \
  --optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-08 --relu-dropout 0.0 --dropout 0.2 \
  --weight-decay 0.0001 --attention-dropout 0.01 \
  --lr-scheduler inverse_sqrt --lr 0.00017 --warmup-updates 4000 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 \
  --bpe fastbpe --bpe-codes $BPECODES \
  --tokenizer moses \
  --decoder-attention-heads 16 --decoder-embed-dim 1024 --decoder-ffn-embed-dim 4096 \
  --decoder-layers 6 --decoder-output-dim 1024 \
  --encoder-attention-heads 16 --encoder-embed-dim 1024 --encoder-ffn-embed-dim 8192 --encoder-layers 6 \
  --share-decoder-input-output-embed \
