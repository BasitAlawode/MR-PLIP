python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=1 --master_port=2224 \
    --use_env retrieval_vid_cvta.py \
    --config ./configs/retrieval_msrvtt_cvta_large.yaml \
    --checkpoint ./cvta_large.pth