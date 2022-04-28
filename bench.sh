#!/bin/bash
export NVIDIA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

docker build --tag knn-bench .

sudo nvidia-smi -i 0 -pm 1

# nvidia-smi -i 0 --query-supported-clocks=mem,gr --format=csv
SELECTED_MEM_CLOCKS=("9751" "9501" "5001")
SELECTED_GPU_CLOCKS=("2100" "1575" "1050")

for memclock in ${SELECTED_MEM_CLOCKS[*]}; do
for gpuclock in ${SELECTED_GPU_CLOCKS[*]}; do
  sudo nvidia-smi -i 0 -ac ${memclock},${gpuclock}
  docker container run --gpus 0 -v /$(pwd)/output:/output knn-bench
done
done

sudo nvidia-smi -i 0 -rac
sudo nvidia-smi -i 0 -pm 0
