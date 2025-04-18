#!/bin/bash
#SBATCH --job-name=0062_bert
#SBATCH --partition=gpu-small
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --wait-all-nodes=1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

nproc

HAED_NODE_IP=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

MAX_SEQ_LEN=1024
GPUS=$((SLURM_NNODES * SLURM_GPUS_PER_NODE))
GRADIENT_ACCUMULATION_STEPS=8
WARMUP_STEPS=24000
LEARNING_RATE=5e-4
PER_DEVICE_BATCH_SIZE=26
MAX_STEPS=500000
TOTAL_BATCH_SIZE=$((GPUS * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
ADAM_BETA1=0.9
ADAM_BETA2=0.98
ADAM_EPSILON=1e-6
MLM_PROBABILITY=0.30
MAX_GRAD_NORM=1.0
BUFFER_SIZE=50000
WEIGHT_DECAY=1e-5

MODEL_NAME=llm-jp-modernbert-base

DATE=`date "+%Y%m%d-%H%M"`
EXP_NAME=${MODEL_NAME}_msl${MAX_SEQ_LEN}-lr${LEARNING_RATE}-bs${TOTAL_BATCH_SIZE}-ws${WARMUP_STEPS}-ms${MAX_STEPS}-mlm${MLM_PROBABILITY}-buf${BUFFER_SIZE}-${DATE}


# Wandb settings
export WANDB_ENTITY=llm-jp
export WANDB_PROJECT=0062_bert

# copy the script to the logs
mkdir -p logs/${EXP_NAME}
cp train.sh logs/${EXP_NAME}/

export LAUNCHER="accelerate launch \
    --config_file config/2node.json \
    --num_processes $GPUS \
    --num_machines $SLURM_NNODES \
    --machine_rank $SLURM_NODEID \
    --main_process_ip $HAED_NODE_IP \
    --rdzv_backend c10d \
    --main_process_port 29500 "

export SCRIPT="src/train/train.py"

export SCRIPT_ARGS=" \
    --model_name_or_path $MODEL_NAME \
    --train_dir llm-jp-corpus-v4-preprocessed-removed \
    --validation_dir llm-jp-corpus-validation/ja \
    --output_dir logs/${EXP_NAME} \
    --num_train_epochs 1 \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --max_seq_length $MAX_SEQ_LEN \
    --learning_rate $LEARNING_RATE \
    --num_warmup_steps $WARMUP_STEPS \
    --adam_beta1 $ADAM_BETA1 \
    --adam_beta2 $ADAM_BETA2 \
    --adam_epsilon $ADAM_EPSILON \
    --max_grad_norm $MAX_GRAD_NORM \
    --mlm_probability $MLM_PROBABILITY \
    --weight_decay $WEIGHT_DECAY \
    --checkpointing_steps 1000 \
    --logging_steps 10 \
    --validation_steps 500 \
    --with_tracking \
    --report_to wandb \
    --project_name $WANDB_PROJECT \
    --exp_name $EXP_NAME \
    --entity $WANDB_ENTITY \
    --max_train_steps $MAX_STEPS \
    --buffer_size $BUFFER_SIZE \
    --preprocessing_num_workers 14 \
    --line_by_line True \
    --resume_from_checkpoint latest \
    "

export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
# export CMD="python $SCRIPT $SCRIPT_ARGS"
srun $CMD