#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# use envs as local overrides for convenience
# e.g.
# NGPU=4 ./run_memory_estimation.sh
NGPU=${NGPU:-"4"}
NNODES=${NNODES:-"32"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

CONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama3_70b.toml"}

BATCH_SIZES=(2 2 2 1 1)
# BATCH_SIZES=(8 8 8 4 4)
SEQ_LENS=(64 256 1024 4096 8192)
AC_MODES=(selective selective full full full)
TP_DEGREE=1
DP_REPLICATE_DEGREE=1
DUMP_FOLDER=/n/holyscratch01/idreos_lab/Users/spurandare/torchtitan/outputs/llama3_70b_FSDP_estimation

# Calculate WORLD_SIZE as the product of NGPU and NNODES
# Export WORLD_SIZE and LOCAL_RANK
export WORLD_SIZE=$((NGPU * NNODES))
export LOCAL_RANK=0
export RANK=0

for ((i=0; i<5; i++)); do

    BATCH_SIZE=${BATCH_SIZES[i]}
    SEQ_LEN=${SEQ_LENS[i]}
    AC_MODE=${AC_MODES[i]}

    python runtime_estimation.py \
    --job.config_file ${CONFIG_FILE} \
    --job.dump_folder ${DUMP_FOLDER} \
    --training.batch_size $BATCH_SIZE \
    --training.seq_len $SEQ_LEN \
    --training.tensor_parallel_degree ${TP_DEGREE} \
    --training.data_parallel_replicate_degree ${DP_REPLICATE_DEGREE} \
    --activation_checkpoint.mode $AC_MODE \
    --memory_estimation.enabled \
    $overrides
done
