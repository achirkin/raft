#!/bin/bash
export NVIDIA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

MEM_CLOCK=$(nvidia-smi -i 0 --query-gpu=clocks.applications.memory --format=csv,noheader,nounits)
GPU_CLOCK=$(nvidia-smi -i 0 --query-gpu=clocks.applications.graphics --format=csv,noheader,nounits)
REPORT_NAME=knn_$(date +"%Y.%m.%d-%H.%M")_${MEM_CLOCK}-${GPU_CLOCK}.csv

./cpp/build/bench_raft \
    --benchmark_filter=KNN/.*/manual_time \
    --benchmark_min_time=10.0 \
    --benchmark_out_format=csv \
    --benchmark_out="/output/${REPORT_NAME}"
