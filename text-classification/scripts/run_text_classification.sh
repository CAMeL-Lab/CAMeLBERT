#!/bin/bash
#SBATCH -p condo
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=120000
# Walltime format hh:mm:ss
#SBATCH --time=11:55:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

nvidia-smi
module purge

#######################################
# Text classification fine-tuning script
#######################################

export ARABIC_DATA=data/train
export TASK_NAME=arabic_sentiment

# export ARABIC_DATA=/scratch/ba63/arabic_poetry_dataset/
# export TASK_NAME=arabic_poetry

# export ARABIC_DATA=/scratch/ba63/MADAR-SHARED-TASK-final-release-25Jul2019/MADAR-Shared-Task-Subtask-1/
# export TASK_NAME=arabic_did_madar_26

# export ARABIC_DATA=/scratch/ba63/MADAR-SHARED-TASK-final-release-25Jul2019/MADAR-Shared-Task-Subtask-1/
# export TASK_NAME=arabic_did_madar_6

# export ARABIC_DATA=/scratch/ba63/MADAR-SHARED-TASK-final-release-25Jul2019/MADAR-Shared-Task-Subtask-2/MADAR-tweets/
# export TASK_NAME=arabic_did_madar_twitter


# export ARABIC_DATA=/scratch/ba63/NADI/NADI_release/
# export TASK_NAME=arabic_did_nadi_country

# aubmindlab/bert-base-arabertv01
# lanwuwei/GigaBERT-v4-Arabic-and-English
# bashar-talafha/multi-dialect-bert-base-arabic
# asafaya/bert-base-arabic
# /scratch/nlp/CAMeLBERT/model/bert-base-wp-30k_msl-512-CA-full-1000000-step
# /scratch/nlp/CAMeLBERT/model/bert-base-wp-30k_msl-512-MSA-full-1000000-step
# /scratch/nlp/CAMeLBERT/model/bert-base-wp-30k_msl-512-MIX-full-1000000-step
# /scratch/nlp/CAMeLBERT/model/bert-base-wp-30k_msl-512-DA-full-1000000-step
# /scratch/nlp/CAMeLBERT/model/bert-base-wp-30k_msl-512-MSA-half-1000000-step
# /scratch/nlp/CAMeLBERT/model/bert-base-wp-30k_msl-512-MSA-quarter-1000000-step
# /scratch/nlp/CAMeLBERT/model/bert-base-wp-30k_msl-512-MSA-eighth-1000000-step
# /scratch/nlp/CAMeLBERT/model/bert-base-wp-30k_msl-512-MSA-sixteenth-1000000-step
# bert-base-multilingual-cased

# /scratch/ba63/UBC-NLP/MARBERT
# /scratch/ba63/UBC-NLP/ARBERT
# /scratch/ba63/bert-base-arabertv02

python run_text_classification.py \
  --model_type bert \
  --model_name_or_path /scratch/nlp/CAMeLBERT/model/bert-base-wp-30k_msl-512-MSA-full-1000000-step \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --eval_all_checkpoints \
  --save_steps 500 \
  --data_dir $ARABIC_DATA \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --overwrite_output_dir \
  --overwrite_cache \
  --output_dir /scratch/ba63/fine_tuned_models/sentiment_models/CAMeLBERT_MSA_arabic_sentiment/$TASK_NAME \
  --seed 12345
