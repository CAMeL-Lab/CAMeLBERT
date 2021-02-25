#!/bin/bash
#SBATCH -p nvidia 
# use gpus
#SBATCH --gres=gpu:1
# memory
#SBATCH --mem=120000
# Walltime format hh:mm:ss
#SBATCH --time=11:30:00
# Output and error files
#SBATCH -o job.%J.out
#SBATCH -e job.%J.err

nvidia-smi
module purge

# export ARABIC_DATA=data/test
# export TASK_NAME=arabic_sentiment
export ARABIC_DATA=/scratch/ba63/arabic_poetry_dataset/
export TASK_NAME=arabic_poetry

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

# export ARABIC_DATA=/scratch/ba63/MADAR-SHARED-TASK-final-release-25Jul2019/MADAR-Shared-Task-Subtask-1/
# export TASK_NAME=arabic_did

python run_text_classification.py \
  --model_type bert \
  --model_name_or_path  /scratch/nlp/CAMeLBERT/model/bert-base-wp-30k_msl-512-MSA-sixteenth-1000000-step \
  --task_name $TASK_NAME \
  --do_eval \
  --eval_all_checkpoints \
  --data_dir $ARABIC_DATA \
  --max_seq_length 128 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 3e-5 \
  --overwrite_cache \
  --output_dir /scratch/ba63/fine_tuned_models/poetry_models/CAMeLBERT_MSA_sixteenth_poetry/$TASK_NAME \
  --seed 12345
