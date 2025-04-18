set -eux

TOKENIZERS_PARALLELISM=false # To avoid warning

MAX_SEQ_LEN=1024
GPUS=4
GRADIENT_ACCUMULATION_STEPS=8
WARMUP_STEPS=1000
LEARNING_RATE=5e-4
PER_DEVICE_BATCH_SIZE=8
MAX_STEPS=100000
TOTAL_BATCH_SIZE=$((GPUS * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS))
ADAM_BETA1=0.9
ADAM_BETA2=0.98
ADAM_EPSILON=1e-6
MLM_PROBABILITY=0.30
MAX_GRAD_NORM=1.0
BUFFER_SIZE=50000
WEIGHT_DECAY=1e-5

MODEL_NAME=llm-jp-modernbert-base

DATASET_NAME=wiki_ja
if [ "$DATASET_NAME" = "wiki_ja" ]; then
    TRAIN_DATA_ARGS=" \
        --dataset_name wikimedia/wikipedia \
        --dataset_config_name 20231101.ja \
    "
elif [ "$DATASET_NAME" = "ja_cc" ]; then
    TRAIN_DATA_ARGS=" \
        --train_dir /model/llm-jp-corpus-public/llm-jp-corpus-v3-public/ja/ja_cc \
    "
fi

DATE=`date "+%Y%m%d-%H%M"`
EXP_NAME=${MODEL_NAME}_${DATASET_NAME}_msl${MAX_SEQ_LEN}-lr${LEARNING_RATE}-bs${TOTAL_BATCH_SIZE}-ws${WARMUP_STEPS}-ms${MAX_STEPS}-mlm${MLM_PROBABILITY}-buf${BUFFER_SIZE}-${DATE}

# Wandb settings
export WANDB_ENTITY=speed-workspace
export WANDB_PROJECT=llm-jp-modernbert

export LAUNCHER="accelerate launch \
    --config_file config/four.json "

export SCRIPT="src/train/train.py"


export SCRIPT_ARGS=" \
    --model_name_or_path $MODEL_NAME \
    ${TRAIN_DATA_ARGS} \
    --validation_dir  dataset/wiki_ja_nano/test \
    --output_dir logs/${EXP_NAME} \
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
    --preprocessing_num_workers 1 \
    --line_by_line True \
    --resume_from_checkpoint latest \
    "

export CMD="$LAUNCHER $SCRIPT $SCRIPT_ARGS"
$CMD