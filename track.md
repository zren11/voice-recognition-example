node0:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --rdzv_id=vr_exp_001 --rdzv_backend=c10d --rdzv_endpoint="172.31.33.72:28080" train.py

node1:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --rdzv_id=vr_exp_001 --rdzv_backend=c10d --rdzv_endpoint="172.31.33.72:28080" train.py
