#!/bin/bash
export NVIDIA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

OUTPUT_DIR="${2:-$(pwd)/knnbench-output}"
case $OUTPUT_DIR= in
  /*) OUTPUT_DIR="`readlink -f -m \"${OUTPUT_DIR}\"`" ;;
  *) OUTPUT_DIR="`readlink -f -m \"$(pwd)/${OUTPUT_DIR}\"`" ;;
esac

mkdir -p $OUTPUT_DIR
chmod 777 $OUTPUT_DIR

docker build --tag knn-bench .

sudo nvidia-smi -i 0 -pm 1

# nvidia-smi -i 0 --query-supported-clocks=mem,gr --format=csv
SELECTED_MEM_CLOCKS=$(nvidia-smi -i 0 --query-supported-clocks=mem --format=csv,noheader,nounits)
SELECTED_GPU_CLOCKS=("1410" "1395" "1380" "1320" "1200" "1110" "1050" "705")


for memclock in ${SELECTED_MEM_CLOCKS[*]}; do
for gpuclock in ${SELECTED_GPU_CLOCKS[*]}; do
  sudo nvidia-smi -i 0 -ac ${memclock},${gpuclock}
  docker container run --gpus 0 -v $OUTPUT_DIR:/output knn-bench
done
done

sudo nvidia-smi -i 0 -rac
sudo nvidia-smi -i 0 -pm 0
