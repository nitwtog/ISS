TASK=sciie
SCALE=small
OUTPUT_DIR=./results
SAVENAME=$TASK-$SCALE-scale
MLM_WEIGHT=20
EXTERNAL_RATIO=999
LR=1e-4
WD=0.01
WARMUP=15000

if [[ $TASK == "imdb" ]]
then
MAXLEN=512
else
MAXLEN=128
fi

mkdir -p $OUTPUT_DIR

accelerate launch --config_file ./accelerate_config/example_config.yaml src/run.py \
    --max_train_steps 150000 \
    --steps_to_eval 10000 \
    --steps_to_save 10000 \
    --steps_to_log 300 \
    --external_dataset_name sample_external.csv \
    --preprocessing_num_workers 32 \
    --max_length $MAXLEN \
    --max_ckpts_to_keep 3 \
    --pad_to_max_length \
    --config_dir yxchar/tlm-${TASK}-${SCALE}-scale \
    --from_scratch \
    --output_dir $OUTPUT_DIR/$SAVENAME \
    --per_device_train_batch_size 128 \
    --per_device_eval_batch_size 64 \
    --cuda_devices 0,1 \
    --task_name $TASK \
    --save_final \
    --mlm_weight $MLM_WEIGHT \
    --external_ratio $EXTERNAL_RATIO \
    --mask_task \
    --weight_decay $WD \
    --learning_rate $LR \
    --num_warmup_steps $WARMUP \
    --dataset_dir /workspace/sciie/TLM-random