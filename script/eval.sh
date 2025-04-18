#!/bin/sh
#SBATCH --job-name=0062_bert
#SBATCH --partition=gpu-debug
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --wait-all-nodes=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# MODEL_NAME=google-bert/bert-base-cased
# MODEL_NAME=FacebookAI/roberta-base
# MODEL_NAME=ku-nlp/deberta-v3-base-japanese
# MODEL_NAME=google-bert/bert-base-multilingual-cased
# MODEL_NAME=studio-ousia/luke-japanese-large-lite
# MODEL_NAME=tohoku-nlp/bert-base-japanese-v3
# MODEL_NAME=sbintuitions/modernbert-ja-130m
# MODEL_NAME=sbintuitions/modernbert-ja-310m
# MODEL_NAME=speed/llm-jp-modernbert-base-stage1
MODEL_NAME=speed/llm-jp-modernbert-base-v4-ja-stage1-15k
# MODEL_NAME=speed/llm-jp-modernbert-base-stage1-100k
# MODEL_NAME=speed/llm-jp-modernbert-base-stage1-200k
# MODEL_NAME=speed/llm-jp-modernbert-base-stage1-300k
# MODEL_NAME=speed/llm-jp-modernbert-base-stage1-400k
# MODEL_NAME=speed/llm-jp-modernbert-base-stage2
# MODEL_NAME=speed/llm-jp-modernbert-base-stage2-filtered
# MODEL_NAME=retrieva-jp/bert-1.3b
# MODEL_NAME=speed/llm-jp-modernbert-base-v4-ja-stage1-300k
# MODEL_NAME=speed/llm-jp-modernbert-base-v4-ja-stage1-500k
# MODEL_NAME=speed/llm-jp-modernbert-base-v3-ja-en-stage1-500k
MODEL_NAME=speed/llm-jp-modernbert-base-v4-ja-mlm0.15-stage1-94k

TASKS=("JSTS" "JNLI" "JCoLA")
LEARNING_RATES=(5e-6 1e-5 2e-5 3e-5 5e-5 1e-4)
EPOCHS=(1 2 3 4 5 10)

OUTPUT_DIR=results

for TASK_NAME in ${TASKS[@]}; do
    for LR in ${LEARNING_RATES[@]}; do
        for EPOCH in ${EPOCHS[@]}; do
            echo "Running task: $TASK_NAME, LR: $LR, Epochs: $EPOCH"

            python src/eval/run_glue_no_trainer.py \
                --model_name_or_path $MODEL_NAME \
                --task_name $TASK_NAME \
                --max_length 512 \
                --per_device_train_batch_size 32 \
                --learning_rate $LR \
                --num_train_epochs $EPOCH \
                --output_dir $OUTPUT_DIR
        done
    done
done
