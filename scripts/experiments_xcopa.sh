#!/bin/bash


# We will first need to fine-tune the model on SIQA dataset
LR=1e-5
EPOCHS=4
MAX_TRAIN_SAMPLES=-1
SEED=42
ALPHA_SMOOTHING=0.1 # For LS
python -m src.run_mcq \
    --dataset siqa \
    --mmlm bert-base-multilingual-uncased \
    --lr $LR \
    --num_epochs $EPOCHS \
    --max_train_samples $MAX_TRAIN_SAMPLES \
    --seed $SEED \
    --alpha_smoothing $ALPHA_SMOOTHING \
    --clip_grad_norm 1

# Now we fine-tune the model on COPA dataset
pretrained_model="siqa_bert-base-multilingual-uncased_MaxLength128_BatchSize8_LR${LR}_EPOCHS${EPOCHS}_SEED${SEED}_TrainSize${MAX_TRAIN_SAMPLES}_MaxGradNorm1_0_AlphaSmoothing${ALPHA_SMOOTHING}_debugFalse"
LR_COPA=3e-5
EPOCHS_COPA=10
python -m src.run_mcq \
    --dataset copa \
    --mmlm bert-base-multilingual-uncased \
    --pretrained_model ${pretrained_model} \
    --lr ${LR_COPA} \
    --num_epochs ${EPOCHS_COPA} \
    --clip_grad_norm 1 \
    --eval_every 10 \
    --merge_dev_sets

# Calibrate on Swahili specifically using it's own dev data (Self-TS + LS)
python -m src.run_mcq \
    --dataset copa \
    --mmlm bert-base-multilingual-uncased \
    --pretrained_model ${pretrained_model} \
    --lr ${LR_COPA} \
    --num_epochs ${EPOCHS_COPA} \
    --clip_grad_norm 1 \
    --merge_dev_sets \
    --temp_scaling \
    --cal_lang sw

# Calibrate on Swahili specifically using it's own dev data for fine-tuning (FSL + LS)
python -m src.run_mcq \
    --dataset copa \
    --mmlm bert-base-multilingual-uncased \
    --pretrained_model ${pretrained_model} \
    --lr ${LR_COPA} \
    --num_epochs ${EPOCHS_COPA} \
    --clip_grad_norm 1 \
    --merge_dev_sets \
    --few_shot_learning \
    --few_shot_lang sw \
    --eval_every 10