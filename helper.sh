# create kind cluster
kind create cluster --name ml --config kind-config.yaml
export VERSION=v2.0.0
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/manager?ref=${VERSION}"
# and wait for it to be ready
k apply -f kubeflow-runtime-example.yaml

# spin up jupyter notebook
jupyter notebook --no-browser --NotebookApp.token='rETIRnArIDEs'


# build docker image
docker build -t speech-recognition-image:0.1 -f Dockerfile .

# load docker image to kind cluster
kind load docker-image speech-recognition-image:0.1 --name ml


# To manually run on 2 nodes
#node0:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --rdzv_id=vr_exp_001 --rdzv_backend=c10d --rdzv_endpoint="<node_0_ip>:28080" train.py

# node1:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 --rdzv_id=vr_exp_001 --rdzv_backend=c10d --rdzv_endpoint="<node_0_ip>2:28080" train.py
