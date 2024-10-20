#!/bin/bash

if [[ $# -lt 1 ]] || [[ $# -gt 1 ]]; then
    echo "Incorrect number of arguments"
    echo "Usage: $0 config_file"
    exit 1
fi

CONFIG_FILE=${1}


edir="${DUMP_DIR}"
ename="${JOB_ID}_v${MAST_HPC_JOB_VERSION}_a${MAST_HPC_JOB_ATTEMPT_INDEX}"
dataset_path="/mnt/mffuse/c4"
save_tb_folder="/mnt/wsfuse/outputs/${JOB_ID}/tb"


echo dump_dir=$edir
echo experiment_name=$ename



LIBCUDA="/usr/local/fbcode/platform010/lib/libcuda.so"
export LIBCUDA_DIR="${LIBCUDA%/*}"
export TRITON_LIBCUDA_PATH="/usr/local/fbcode/platform010/lib/"
export LD_PRELOAD="${PRELOAD_PATH:=$LIBCUDA:/usr/local/fbcode/platform010/lib/libnvidia-ml.so}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_DIR}/lib"
export PYTHONPATH="$PYTHONPATH:$TORCHX_RUN_PYTHONPATH"

source ${CONDA_DIR}/bin/activate

cd /packages/torchtitan_additional_packages/torchtitan


###############
#  do whatever you like below
###############


BATCH_SIZES=(2 2 2 4)
SEQ_LENS=(128 512 2048 8192)
AC_MODES=(selective selective full full)

for ((i=0; i<4; i++)); do
  BATCH_SIZE=${BATCH_SIZES[i]}
  SEQ_LEN=${SEQ_LENS[i]}
  AC_MODE=${AC_MODES[i]}

  PYTORCH_KERNEL_CACHE_PATH="/mnt/mffuse/.cache/torch/kernels" \
    TORCH_DISABLE_ADDR2LINE=1 \
    python train.py \
    --job.config_file "${CONFIG_FILE}" \
    --job.dump_folder "${edir}" \
    --training.dataset_path "${dataset_path}" \
    --metrics.save_tb_folder "${save_tb_folder}" \
    --training.batch_size $BATCH_SIZE \
    --training.seq_len $SEQ_LEN \
    --activation_checkpoint.mode $AC_MODE \
    --profiling.enable_cuda_event_iter_time

  sleep 5

  PYTORCH_KERNEL_CACHE_PATH="/mnt/mffuse/.cache/torch/kernels" \
    TORCH_DISABLE_ADDR2LINE=1 \
    python train.py \
    --job.config_file "${CONFIG_FILE}" \
    --job.dump_folder "${edir}" \
    --training.dataset_path "${dataset_path}" \
    --metrics.save_tb_folder "${save_tb_folder}" \
    --training.batch_size $BATCH_SIZE \
    --training.seq_len $SEQ_LEN \
    --activation_checkpoint.mode $AC_MODE \
    --profiling.enable_cuda_event_iter_time \
    --comm.enable_fake_pg

  sleep 5

  PYTORCH_KERNEL_CACHE_PATH="/mnt/mffuse/.cache/torch/kernels" \
    TORCH_DISABLE_ADDR2LINE=1 \
    python train.py \
    --job.config_file "${CONFIG_FILE}" \
    --job.dump_folder "${edir}" \
    --training.dataset_path "${dataset_path}" \
    --metrics.save_tb_folder "${save_tb_folder}" \
    --training.batch_size $BATCH_SIZE \
    --training.seq_len $SEQ_LEN \
    --activation_checkpoint.mode $AC_MODE \
    --profiling.enable_profiling \
    --metrics.enable_tensorboard
done