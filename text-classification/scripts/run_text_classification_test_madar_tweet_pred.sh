#!/bin/bash
#SBATCH -p nvidia 
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

###################################
# MADAR-Twitter-5 Test Eval script
###################################


# export ARABIC_DATA=data/train
# export TASK_NAME=arabic_sentiment

# export ARABIC_DATA=/scratch/ba63/arabic_poetry_dataset/
# export TASK_NAME=arabic_poetry

# export ARABIC_DATA=/scratch/ba63/MADAR-SHARED-TASK-final-release-25Jul2019/MADAR-Shared-Task-Subtask-1/
# export TASK_NAME=arabic_did_madar_26

# export ARABIC_DATA=/scratch/ba63/MADAR-SHARED-TASK-final-release-25Jul2019/MADAR-Shared-Task-Subtask-1/
# export TASK_NAME=arabic_did_madar_6

export ARABIC_DATA=/scratch/ba63/MADAR-SHARED-TASK-final-release-25Jul2019/MADAR-Shared-Task-Subtask-2/MADAR-tweets/
export TASK_NAME=arabic_did_madar_twitter


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

export OUTPUT_DIR=/scratch/ba63/fine_tuned_models/did_models_MADAR_twitter/CAMeLBERT_MSA_sixteenth_DID/$TASK_NAME/checkpoint-10500-best/

python run_text_classification.py \
  --model_type bert \
  --model_name_or_path /scratch/nlp/CAMeLBERT/model/bert-base-wp-30k_msl-512-MSA-sixteenth-1000000-step \
  --task_name $TASK_NAME \
  --do_pred \
  --write_preds \
  --data_dir $ARABIC_DATA \
  --max_seq_length 128 \
  --per_gpu_eval_batch_size 32 \
  --overwrite_cache \
  --output_dir $OUTPUT_DIR \
  --seed 12345

export test_user_ids=/scratch/ba63/MADAR-SHARED-TASK-final-release-25Jul2019/MADAR-Shared-Task-Subtask-2/MADAR-tweets/grouped_tweets.test.users
export preds=$OUTPUT_DIR/predictions.txt

paste -d '\t' $test_user_ids $preds > $OUTPUT_DIR/users_and_preds

python utils/vote_did.py --preds_file_path $OUTPUT_DIR/users_and_preds --output_file_path $OUTPUT_DIR/users_and_preds.voting


python /scratch/ba63/MADAR-SHARED-TASK-final-release-25Jul2019/MADAR-DID-Scorer.py  /scratch/ba63/MADAR-SHARED-TASK-final-release-25Jul2019/MADAR-Shared-Task-Subtask-2/MADAR-tweets/test.gold.labels $OUTPUT_DIR/users_and_preds.voting > $OUTPUT_DIR/test_results.voting.txt
