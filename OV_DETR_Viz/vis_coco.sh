python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --use_env inference.py \
    --coco_path <dataset_path> \
    --num_queries 300 \
    --label_map \
    --with_box_refine \
    --two_stage \
    --eval \
    --resume <checkpoint_path> \
