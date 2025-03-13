
CUDA_VISIBLE_DEVICES=0 fairseq-train \
	--user-dir ../../graphormer \
	--batch-size 64 \
	--batch-size 64 \
	--num-workers 20 \
	--ddp-backend=legacy_ddp \
	--seed 23 \
	--user-data-dir RP_training_dataset \
	--dataset-name RT_Library \
	--task graph_prediction_with_flag \
	--criterion rmse \
	--arch graphormer_base \
	--num-classes 1 \
	--attention-dropout 0.15 --act-dropout 0.10 --dropout 0.10 \
	--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.01 \
	--lr-scheduler polynomial_decay --power 1 --warmup-updates 33281 --total-num-update 221875\
	--lr 1e-4 \
	--fp16 \
    --encoder-layers 8 \
    --encoder-embed-dim  512 \
    --encoder-ffn-embed-dim 512 \
    --encoder-attention-heads 64 \
	--fp16-scale-tolerance 0.05 \
    --mlp-layers 5 \
	--max-epoch 250 \
	--no-epoch-checkpoints \
	--freeze-level 0 \
	--save-dir ../../checkpoints \


