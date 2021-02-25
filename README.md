# The Interplay of Variant, Size, and Task Type in Arabic Pre-trained Language Models:

This repo contains code for the experiments in our paper [The Interplay of Variant, Size, and Task Type in Arabic Pre-trained Language Models]().

# Requirements:

This code was written for python>=3.7, pytorch 1.5.1, and transformers 3.1.0. You will need a few additional packages. Here's how you can set up the environment using conda (assuming you have conda and cuda installed):

```bash
git clone https://github.com/balhafni/CAMeLBERT.git
cd CAMeLBERT

conda create -n CAMeLBERT python=3.7
conda activate CAMeLBERT

pip install -r requirements.txt
```

# Fine-tuning Experiments:

## Text Classification:

### Sentiment Analysis:

For the sentiment analysis experiments, we combined four datasets: 1) [ArSAS](); 2) [ASTD](); 3) [SemEval-2017 4A](); 4) [ArSenTD]().</br>
The models were fine-tuned on ArSenTD and the train splits of ArSAS, ASTD, and SemEval-2017. We then evaluate all the checkpoints on 
a single dev split from ArSAS, ASTD, and SemEval-2017 and pick the best checkpoint to report the results on the test splits of ArSAS, ASTD, and SemEval-2017 repsectively. To run the fine-tuning:

```bash
export DATA_DIR=/path/to/data
export TASK_NAME=arabic_sentiment

python run_text_classification.py \
  --model_type bert \
  --model_name_or_path /path/to/pretrained_model/  # Or huggingface model id \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --eval_all_checkpoints \
  --save_steps 500 \
  --data_dir $DATA_DIR \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --overwrite_output_dir \
  --overwrite_cache \
  --output_dir /path/to/output_dir \
  --seed 12345
```

### Dialect Identification:

For the dialect identification experiments, we fine-tuned the models on four different dialect identification datasets: 1) [MADAR Corpus 26](); 2) [MADAR Corpus 6](); 3) [MADAR Twitter-5](); 4) [NADI Country-level](). We fine-tuned the models across the four datasets and we report results of the best checkpoints on the dev splits. To run the fine-tuning:


```bash
export DATA_DIR=/path/to/data
export TASK_NAME=arabic_did_madar_26 # or arabic_did_madar_6, arabic_did_madar_twitter, arabic_did_nadi_country

python run_text_classification.py \
  --model_type bert \
  --model_name_or_path /path/to/pretrained_model/  # Or huggingface model id \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --eval_all_checkpoints \
  --save_steps 500 \
  --data_dir $DATA_DIR \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 10.0 \
  --overwrite_output_dir \
  --overwrite_cache \
  --output_dir /path/to/output_dir \
  --seed 12345
```

### Poetry Classification:

For the poetry classification experiments, we fine-tuned the [APCD]() dataset and report the results of the best checkpoints on the dev split. To run the fine-tuning:

```bash
export DATA_DIR=/path/to/data
export TASK_NAME=arabic_poetry

python run_text_classification.py \
  --model_type bert \
  --model_name_or_path /path/to/pretrained_model/  # Or huggingface model id \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --eval_all_checkpoints \
  --save_steps 5000 \
  --data_dir $DATA_DIR \
  --max_seq_length 128 \
  --per_gpu_train_batch_size 32 \
  --per_gpu_eval_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 3.0 \
  --overwrite_output_dir \
  --overwrite_cache \
  --output_dir /path/to/output_dir \
  --seed 12345
```

## Token Classification:

### NER:

For the NER experiments, we used the [ANERCorp]() dataset and followed the splits defined by [Obeid et al., 2020]().
The dataset doesn't have a dev split, so we fine-tune the models on the train split and evaluate the last checkpoint on the test split.
To run the fine-tuning:


```bash
export DATA_DIR=/path/to/data                 # Should contain train/dev/test/labels files
export MAX_LENGTH=500
export BERT_MODEL=/path/to/pretrained_model/  # Or huggingface model id
export OUTPUT_DIR=/path/to/output_dir
export BATCH_SIZE=32
export NUM_EPOCHS=3
export SAVE_STEPS=750
export SEED=12345

 python run_token_classification.py \
  --data_dir $DATA_DIR \
  --labels $DATA_DIR/labels.txt \
  --model_name_or_path $BERT_MODEL \
  --output_dir $OUTPUT_DIR \
  --max_seq_length  $MAX_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --per_device_train_batch_size $BATCH_SIZE \
  --save_steps $SAVE_STEPS \
  --seed $SEED \
  --overwrite_output_dir \
  --overwrite_cache \
  --do_train \
  --do_predict
```

### POS Tagging:

For the POS tagging experiments, we fine-tuned the models on three different datasets:<br/>

1. Penn Arabic Treebank (PATB) ([Maamouri et al., 2004]()): in MSA and has 32 POS tags
2. Egyptian Arabic Treebank (ARZATB) ([Maamouri et al., 2012]()): in EGY and has 33 POS tags
3. GUMAR corpus ([Khalifa et al., 2018]()): in GLF and includes 35 POS tags

We used the same hyperparameters for the 3 datasets. To run the fine-tuning:

```bash
export DATA_DIR=/path/to/data                 # Should contain train/dev/test/labels files
export MAX_LENGTH=512
export BERT_MODEL=/path/to/pretrained_model/  # Or huggingface model id
export OUTPUT_DIR=/path/to/output_dir
export BATCH_SIZE=32
export NUM_EPOCHS=10
export SAVE_STEPS=500
export SEED=12345

python run_token_classification.py \
  --data_dir $DATA_DIR \
  --labels $DATA_DIR/labels.txt \
  --model_name_or_path $BERT_MODEL \
  --output_dir $OUTPUT_DIR \
  --max_seq_length  $MAX_LENGTH \
  --num_train_epochs $NUM_EPOCHS \
  --per_device_train_batch_size $BATCH_SIZE \
  --save_steps $SAVE_STEPS \
  --seed $SEED \
  --overwrite_output_dir \
  --overwrite_cache \
  --do_train \
  --do_eval
```
After that, we run the evaluation on all the checkpoints and pick the one that has the best performance on the dev set; take a look at [run_pos_tagging_pred.sh]().
