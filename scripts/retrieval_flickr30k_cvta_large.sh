python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=1 --master_port=2224 \
    --use_env retrieval_img_cvta.py \
    --config ./configs/retrieval_flickr30k_cvta_large.yaml \
    --output_dir output/retrieval_flickr30k_cvta_large \
    --checkpoint ./cvta_large.pth \
    --do_two_optim \
    --do_amp