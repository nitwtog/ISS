python train_ft_mlm.py \
    --model_name bert-base-uncased \
    --mode train  \
    --ckpt_dir ./saved_models \
    --task_name citation_intent \
    --mlm_weight 3 \
    --max_seq_length 128 \
    --bsz 16 \
    --epochs 15 \
    --lr 5e-5 \
    --weight_decay 0 \
    --modeldir citation_intent \
    --warmup_ratio 0.1 \
    --cuda 1 \
    --seed 42

