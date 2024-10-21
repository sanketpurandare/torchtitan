#!/usr/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

torchx run mast.py:train \
    --additional_folders ~/local/torchtitan \
    --twtask_bootstrap_script run_torchtitan_debug_model.sh \
    --name llama3_70b_bench \
    --h grandteton \
    --nodes 2  \
    train_configs/debug_model.toml

sleep 2

torchx run mast.py:train \
    --additional_folders ~/local/torchtitan \
    --twtask_bootstrap_script run_torchtitan_llama3_70b.sh \
    --name llama3_70b_bench \
    --h grandteton \
    --nodes 16  \
    train_configs/llama3_70b.toml

sleep 2

torchx run mast.py:train \
    --additional_folders ~/local/torchtitan \
    --twtask_bootstrap_script run_torchtitan_llama3_405b.sh \
    --name llama3_405b_bench \
    --h grandteton \
    --nodes 32  \
    train_configs/llama3_405b.toml

