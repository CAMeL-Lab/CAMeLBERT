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
# /scratch/ba63/bert-base-arabertv02/

export DATA_DIR=/scratch/ba63/magold_files/EGY
export MAX_LENGTH=512
export BERT_MODEL=/scratch/nlp/CAMeLBERT/model/bert-base-wp-30k_msl-512-MSA-full-1000000-step
export OUTPUT_DIR=/scratch/ba63/fine_tuned_models/pos_models/EGY/CAMeLBERT_MSA_POS_EGY
export BATCH_SIZE=32
export SAVE_STEPS=500
export SEED=12345

for f in $OUTPUT_DIR/checkpoint-*/

do
echo $f
python run_token_classification.py \
--data_dir $DATA_DIR \
--labels $DATA_DIR/labels.txt \
--model_name_or_path $f \
--output_dir $f \
--max_seq_length  $MAX_LENGTH \
--per_gpu_eval_batch_size $BATCH_SIZE \
--seed $SEED \
--overwrite_cache \
--do_eval
done
