# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import gc
from pathlib import Path
import os
import time

import torch
from torch._guards import active_fake_mode
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed._tools.fsdp2_mem_tracker import FSDPMemTracker
from torch.distributed._tools.spmd_runtime_estimator import SPMDRuntimeEstimator
from torch.distributed._tools.fake_collectives import CollDistMode
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_hf_data_loader, build_tokenizer
from torchtitan.float8 import Float8Handler
from torchtitan.logging import init_logger, logger
from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config
from torchtitan.optimizer import build_lr_schedulers, build_optimizers
from torchtitan.parallelisms import models_parallelize_fns, ParallelDims
from train import get_train_context


def estimate_runtime(job_config: JobConfig):
    init_logger()
    logger.info("Estimating runtime...")
    gc.disable()
    gc.collect(1)

    # Get the world size
    world_size = int(os.environ["WORLD_SIZE"])

    # if tp > or pp > 1, we exit
    if (
        job_config.experimental.pipeline_parallel_degree > 1
    ):
        logger.info(
            "Pipeline parallelism are not supported yet."
        )
        return

    # fake tensor doesn't work with fused rmsnorm
    if (
        job_config.model.norm_type == "fused_rmsnorm"
        and not job_config.memory_estimation.disable_fake_mode
    ):
        logger.info(
            "Fused RMSNorm is not supported yet under fake estimation mode. "
            "Switching to rmsnorm."
        )
        job_config.model.norm_type = "rmsnorm"

    if job_config.model.norm_type == "compiled_rmsnorm":
        logger.info("Compiled RMSNorm is not supported yet. Switching to RMSNorm.")
        job_config.model.norm_type = "rmsnorm"

    if job_config.training.compile or job_config.experimental.enable_compiled_autograd:
        logger.info("Compile mode is not supported yet. Switching to eager mode.")
        job_config.training.compile = False
        job_config.experimental.enable_compiled_autograd = False

    parallel_dims = ParallelDims(
        dp_shard=job_config.training.data_parallel_shard_degree,
        dp_replicate=job_config.training.data_parallel_replicate_degree,
        tp=job_config.training.tensor_parallel_degree,
        pp=job_config.experimental.pipeline_parallel_degree,
        world_size=world_size,
        enable_loss_parallel=job_config.training.enable_loss_parallel,
    )

    device = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(device)

    # init fake pg
    store = FakeStore()
    torch.distributed.init_process_group(
        "fake", rank=int(os.environ["LOCAL_RANK"]), world_size=world_size, store=store, group_name="custom"
    )

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type="cuda")


    if parallel_dims.dp_enabled:
        if parallel_dims.dp_replicate_enabled:
            dp_mesh = world_mesh["dp_replicate", "dp_shard"]
        else:
            dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
    else:
        dp_degree, dp_rank = 1, 0
        dp_mesh = None

    if not parallel_dims.dp_enabled:
        logger.info("Data parallelism is not enabled. Skipping memory estimation.")
        return

    model_name = job_config.model.name
    if parallel_dims.tp_enabled:
        tp_mesh = world_mesh["tp"]
    else:
        tp_mesh = None

    # build tokenizer
    tokenizer_type = model_name_to_tokenizer[model_name]
    tokenizer = build_tokenizer(tokenizer_type, job_config.model.tokenizer_path)

    # # build dataloader
    # data_loader = build_hf_data_loader(
    #     job_config.training.dataset,
    #     job_config.training.dataset_path,
    #     tokenizer,
    #     job_config.training.batch_size,
    #     job_config.training.seq_len,
    #     dp_degree,
    #     dp_rank,
    # )

    train_context = get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
    )

    # loss fn can be shared by pipeline-parallel or non-pp execution
    def loss_fn(pred, labels):
        return torch.nn.functional.cross_entropy(
            pred.flatten(0, 1), labels.flatten(0, 1)
        )

    # build model (using meta init)
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][job_config.model.flavor]
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. vocab size from tokenizer
    # 3. max_seq_len base on inputs
    model_config.norm_type = job_config.model.norm_type
    model_config.vocab_size = tokenizer.n_words
    model_config.max_seq_len = job_config.training.seq_len

    with FakeTensorMode(allow_non_fake_inputs=True) if not job_config.memory_estimation.disable_fake_mode else contextlib.nullcontext():

        logger.info(
            f"Building {model_name} {job_config.model.flavor} with {model_config}"
        )
        if job_config.memory_estimation.disable_fake_mode:
            with torch.device("meta"):
                model = model_cls.from_model_args(model_config)
        else:
            with torch.device(device):
                model = model_cls.from_model_args(model_config)

        # a no-op hander if float8 is not enabled
        float8_handler = Float8Handler(job_config, parallel_dims)
        # swap to Float8Linear based on float8 configs
        float8_handler.convert_to_float8_training(model)

        # apply PT-D DP/TP parallelisms and activation checkpointing
        models_parallelize_fns[model_name](model, world_mesh, dp_mesh, tp_mesh, parallel_dims, job_config)
        if job_config.memory_estimation.disable_fake_mode:
            model.to_empty(device="cuda")
            model.init_weights()

        model.train()

        # build optimizer after applying parallelisms to the model
        optimizers = build_optimizers([model], job_config)
        lr_schedulers = build_lr_schedulers(optimizers.optimizers, job_config)

        setup_config = {}
        setup_config["batch_size"] = job_config.training.batch_size
        setup_config["seq_len"] = model_config.max_seq_len
        setup_config["ac_mode"] = job_config.activation_checkpoint.mode

        logger.info(f"Vocab size: {model_config.vocab_size}")
        # Create a dummy batch instead of loading from a dataset
        batch = (
            torch.randint(
                0,
                model_config.vocab_size,
                (job_config.training.batch_size, model_config.max_seq_len),
                device="cuda",
            ),
            torch.randint(
                0,
                model_config.vocab_size,
                (job_config.training.batch_size, model_config.max_seq_len),
                device="cuda",
            ),
        )
        spmd_estimator = SPMDRuntimeEstimator(world_mesh)

        def train_step():
            input_ids, labels = batch
            # train step
            with train_context():
                pred = model(input_ids)
                loss = loss_fn(pred, labels)
                del pred
                loss.backward()

            # clip gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), job_config.training.max_norm, foreach=True
            )
            # sync float8 amaxes and scales
            float8_handler.sync_float8_amax_and_scale_history(model)
            # optimizer step
            optimizers.step()
            lr_schedulers.step()
            # calculate float8 dynamic amax/scale for all-parameter for FSDP2
            # it issues a single all-reduce for all parameters at once for better performance
            float8_handler.precompute_float8_dynamic_scale_for_fsdp(model)
            optimizers.zero_grad()
            gc.collect(1)

        train_step()

        with spmd_estimator(estimate_mode_type='operator-level-learned-model', collective_mode_type='learned-model'):
            start_time = time.time()
            train_step()
            end_time = time.time()
                

        # fsdp_memtracker.display_modulewise_snapshots(
        #     depth=3, units="MiB", tabulate=True
        # )
        setup_config["total_runtime"] = spmd_estimator.total_runtime
        setup_config["total_compute"] = spmd_estimator.total_runtime
        setup_config["total_comm"] = spmd_estimator.total_runtime
        setup_config["est_time"] = (end_time - start_time)
       
        file_name = f"{job_config.model.name}_{job_config.model.flavor}"
        if job_config.training.tensor_parallel_degree > 1:
            file_name += f"_2D"
        else:
            file_name += f"_1D"

        rank = int(os.environ['RANK'])
        world_size = int(os.environ["WORLD_SIZE"])
        out_file = f"{file_name}_{world_size}_{rank}_runtime_estimate.txt"
        out_dir = Path(job_config.job.dump_folder)
        out_dir.mkdir(parents=True, exist_ok=True)
        fout = open(f"{out_dir}/{out_file}", "a")
        fout.write(f'{setup_config["batch_size"]},{setup_config["seq_len"]},{setup_config["ac_mode"]},{setup_config["total_runtime"]},{setup_config["total_compute"]},{setup_config["total_comm"]},{setup_config["est_time"]}\n')
        fout.flush()
        fout.close()

        gc.enable()


if __name__ == "__main__":
    config = JobConfig()
    config.parse_args()
    try:
        with CollDistMode():
            estimate_runtime(config)
    finally:
        torch.distributed.destroy_process_group()
