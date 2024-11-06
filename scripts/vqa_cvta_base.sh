# deepspeed==0.5.8

python -m torch.distributed.launch --nproc_per_node=8 \
    --nnodes=1 --master_port=3224 \
    --use_env vqa_cvta.py \
    --config ./configs/vqa_cvta_base.yaml \
    --checkpoint ./cvta_base.pth \
    --output_dir output/vqa_cvta_base \
    --do_two_optim \
    --add_object \
    --max_input_length 80 \
    --do_amp \
    --add_ocr \
    --deepspeed \
    --deepspeed_config ./configs/ds_config.json 
