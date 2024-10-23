#!/bin/bash

#SBATCH --job-name=torchtitan
#SBATCH --partition=kempner_h100
#SBATCH --account=kempner_sham_lab
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --time=03:00:00
#SBATCH --mem=64gb
#SBATCH --output=/n/holyscratch01/idreos_lab/Users/spurandare/torchtitan/job_logs/%x_%j.out
#SBATCH --error=/n/holyscratch01/idreos_lab/Users/spurandare/torchtitan/job_logs/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --chdir=/n/holyscratch01/idreos_lab/Users/spurandare/torchtitan

scontrol show job $SLURM_JOB_ID

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)
echo Node IP: $head_node_ip

export NCCL_DEBUG=INFO

GPUS_PER_NODE=4
echo "GPUS_PER_NODE: $GPUS_PER_NODE"
echo "SLURM_NNODES: $SLURM_NNODES"

CONFIG_FILE=${CONFIG_FILE:-"./train_configs/llama3_70b.toml"}

# BATCH_SIZES=(2 2 2 1 1)
BATCH_SIZES=(8 8 8 4 4)
SEQ_LENS=(64 256 1024 4096 8192)
AC_MODES=(selective selective full full full)
TP_DEGREE=4
DP_REPLICATE_DEGREE=1
DUMP_FOLDER=/n/holyscratch01/idreos_lab/Users/spurandare/torchtitan/outputs/llama3_70b_FSDP_TP2

for ((i=0; i<5; i++)); do
  BATCH_SIZE=${BATCH_SIZES[i]}
  SEQ_LEN=${SEQ_LENS[i]}
  AC_MODE=${AC_MODES[i]}

  srun --output job_logs/%x_%j_%t.out --error job_logs/%x_%j_%t.err torchrun \
        --nnodes ${SLURM_NNODES} \
        --nproc_per_node ${GPUS_PER_NODE} \
        --rdzv_id 101 \
        --rdzv_backend c10d \
        --rdzv_endpoint "${head_node_ip}:29500" \
        ./train.py \
        --job.config_file ${CONFIG_FILE} \
        --job.dump_folder ${DUMP_FOLDER} \
        --training.batch_size $BATCH_SIZE \
        --training.seq_len $SEQ_LEN \
        --training.tensor_parallel_degree ${TP_DEGREE} \
        --training.data_parallel_replicate_degree ${DP_REPLICATE_DEGREE} \
        --activation_checkpoint.mode $AC_MODE \
        --profiling.enable_cuda_event_iter_time
 
  sleep 5
  srun --output job_logs/%x_%j_%t.out --error job_logs/%x_%j_%t.err torchrun \
        --nnodes ${SLURM_NNODES} \
        --nproc_per_node ${GPUS_PER_NODE} \
        --rdzv_id 101 \
        --rdzv_backend c10d \
        --rdzv_endpoint "${head_node_ip}:29500" \
        ./train.py \
        --job.config_file ${CONFIG_FILE} \
        --job.dump_folder ${DUMP_FOLDER} \
        --training.batch_size $BATCH_SIZE \
        --training.seq_len $SEQ_LEN \
        --training.tensor_parallel_degree ${TP_DEGREE} \
        --training.data_parallel_replicate_degree ${DP_REPLICATE_DEGREE} \
        --activation_checkpoint.mode $AC_MODE \
        --profiling.enable_cuda_event_iter_time \
        --comm.enable_fake_pg
 
  sleep 5
  srun --output job_logs/%x_%j_%t.out --error job_logs/%x_%j_%t.err torchrun \
        --nnodes ${SLURM_NNODES} \
        --nproc_per_node ${GPUS_PER_NODE} \
        --rdzv_id 101 \
        --rdzv_backend c10d \
        --rdzv_endpoint "${head_node_ip}:29500" \
        ./train.py \
        --job.config_file ${CONFIG_FILE} \
        --job.dump_folder ${DUMP_FOLDER} \
        --training.batch_size $BATCH_SIZE \
        --training.seq_len $SEQ_LEN \
        --training.tensor_parallel_degree ${TP_DEGREE} \
        --training.data_parallel_replicate_degree ${DP_REPLICATE_DEGREE} \
        --activation_checkpoint.mode $AC_MODE \
        --profiling.enable_profiling \
        --metrics.enable_tensorboard

done
