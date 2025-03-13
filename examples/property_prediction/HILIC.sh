CUDA_VISIBLE_DEVICES=0 fairseq-train \
	--user-dir ../../graphormer \
	--batch-size 64 \
	--batch-size 64 \
	--num-workers 20 \
	--ddp-backend=legacy_ddp \
	--seed 23 \
	--user-data-dir HILIC_train \
	--dataset-name RT_Library \
	--task graph_prediction_with_flag \
	--criterion rmse_HILIC \
	--arch graphormer_HILIC \
	--num-classes 1 \
	--attention-dropout 0.10 --act-dropout 0.10 --dropout 0.10 \
	--optimizer adam --adam-betas '(0.9, 0.999)' --adam-eps 1e-8 --clip-norm 5.0 --weight-decay 0.01 \
	--lr-scheduler polynomial_decay --power 1 --warmup-updates 59 --total-num-update  391 --min-loss-scale 1e-9 \
	--lr 1.5e-4 \
	--fp16 \
    --encoder-layers 8 \
	--wandb-project 'RT_preds' \
    --encoder-embed-dim  512 \
    --encoder-ffn-embed-dim 512 \
    --encoder-attention-heads 64 \
    --mlp-layers 5 \
	--max-epoch 250 \
	--no-epoch-checkpoints \
	--save-dir ../../checkpoints \