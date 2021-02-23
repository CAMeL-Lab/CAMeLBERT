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

## Token Classification:

### NER:


### POS Tagging:

For the POS tagging experiments, we fine-tuned the models on three different datasets:<br/>

1. Penn Arabic Treebank (PATB) ([Maamouri et al., 2004]()): MSA and has 32 POS tags
2. Egyptian Arabic Treebank (ARZATB) ([Maamouri et al., 2012]()): EGY and has 33 POS tags
3. GUMAR corpus ([Khalifa et al., 2018]()): in GLF and includes 35 POS tags

We used the same hyperparameters for the 3 datasets. To run the fine-tuning, you would need to run:

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
