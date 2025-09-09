# create kind cluster
kind create cluster --name ml --config kind-config.yaml
export VERSION=v2.0.0
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/manager?ref=${VERSION}"
# and wait for it to be ready
k apply -f kubeflow-runtime-example.yaml

# spin up jupyter notebook
jupyter notebook --no-browser --NotebookApp.token='rETIRnArIDEs'


# build docker image
docker build -t speech-recognition-image:0.4 -f Dockerfile .

# load docker image to kind cluster
kind load docker-image speech-recognition-image:0.4 --name ml


