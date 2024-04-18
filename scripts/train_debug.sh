#!/bin/bash
# WORK_DIR=$(cd `dirname $0`; pwd)/..

# echo "${@:1}"

# export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=1 # activate InfiniBand
export NCCL_IB_GID_INDEX=0
# export NCCL_SOCKET_IFNAME=eth0 # NCCL Ethernet Port
export NCCL_DEBUG=WARN


# cd $WORK_DIR

export export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=8

LOG_DIR="logs/$1"

torchrun \
--nnodes 1 \
--node_rank 0 \
--nproc_per_node 2 \
mot.py \
--save_log --output_dir $LOG_DIR \
--tracker_name ByteTracker \
--options use_ema=False dataset_file=debug_dataset config_file=config/TBD/single_class.py

# The configuration of dataset_file is specified in the file of motlib/mot_dataset/dataset_config.