#!/bin/bash
export NVIDIA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

GPU_SPEC=$(nvidia-smi -i 0 --query-gpu=name,clocks.applications.memory,clocks.applications.graphics --format=csv,noheader,nounits)
GPU_SPEC=${GPU_SPEC//, /-}
GPU_SPEC=${GPU_SPEC//,/-}
GPU_SPEC=${GPU_SPEC// /-}
REPORT_NAME=knn_$(date +"%Y.%m.%d-%H.%M")_${GPU_SPEC}.csv

./cpp/build/bench_raft \
    --benchmark_filter=KNN/.*/manual_time \
    --benchmark_min_time=10.0 \
    --benchmark_out_format=csv \
    --benchmark_out="/output/${REPORT_NAME}"
