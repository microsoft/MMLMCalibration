#!/bin/bash

# Running vanilla training (wo calibration) for mBERT
echo "Running vanilla training (wo calibration) for mBERT"
python -m src.run_sentence_cls \
    --mmlm bert-base-multilingual-uncased \
    --dataset xnli \
    --lr 3e-5 \
    --num_epochs 3 \
    --max_train_samples 40000

# Running vanilla training (wo calibration) for XLM-R
echo "Running vanilla training (wo calibration) for XLM-R"
python -m src.run_sentence_cls \
    --mmlm xlm-roberta-large \
    --dataset xnli \
    --lr 7e-6 \
    --num_epochs 3 \
    --max_train_samples 40000

# Run experiments with Label Smoothing (LS)
echo "Running experiments with Label Smoothing (LS)"
python -m src.run_sentence_cls \
    --mmlm bert-base-multilingual-uncased \
    --dataset xnli \
    --lr 3e-5 \
    --num_epochs 3 \
    --max_train_samples 40000 \
    --alpha_smoothing 0.1

# Run experiments with Temperature Scaling using English Dev Data (TS)
echo "Running experiments with Temperature Scaling using English Dev Data (TS)"
python -m src.run_sentence_cls \
    --mmlm bert-base-multilingual-uncased \
    --dataset xnli \
    --lr 3e-5 \
    --num_epochs 3 \
    --max_train_samples 40000 \
    --temp_scaling \
    --cal_lang en \
    --cal_size 500

# Combine the two (TS + LS)
echo "Running experiments with TS + LS"
python -m src.run_sentence_cls \
    --mmlm bert-base-multilingual-uncased \
    --dataset xnli \
    --lr 3e-5 \
    --num_epochs 3 \
    --max_train_samples 40000 \
    --alpha_smoothing 0.1 \
    --temp_scaling \
    --cal_lang en \
    --cal_size 500

# Calibrate on Swahili specifically using it's own dev data (Self-TS + LS)
echo "Calibrate on Swahili specifically using it's own dev data (Self-TS + LS)"
python -m src.run_sentence_cls \
    --mmlm bert-base-multilingual-uncased \
    --dataset xnli \
    --lr 3e-5 \
    --num_epochs 3 \
    --max_train_samples 40000 \
    --alpha_smoothing 0.1 \
    --temp_scaling \
    --cal_lang sw \
    --cal_size -1 #-1 means using the entire validation data

# Calibrate on Swahili specifically using it's own dev data for fine-tuning (FSL + LS)
echo "Calibrate on Swahili specifically using it's own dev data for fine-tuning (FSL + LS)"
python -m src.run_sentence_cls \
    --mmlm bert-base-multilingual-uncased \
    --dataset xnli \
    --lr 3e-5 \
    --num_epochs 3 \
    --max_train_samples 40000 \
    --alpha_smoothing 0.1 \
    --few_shot_learning \
    --few_shot_lang sw \
    --few_shot_size -1 #-1 means using the entire validation data