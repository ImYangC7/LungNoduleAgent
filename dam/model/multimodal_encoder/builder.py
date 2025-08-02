# Copyright 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/haotian-liu/LLaVA/
import torch # noqa
import os
from transformers import AutoConfig, PretrainedConfig, PreTrainedModel
from .siglip_encoder import SiglipVisionTower
from .context_provider import ContextProvider, ContextProviderConfig

def build_vision_tower(
    model_name_or_path: str, config: PretrainedConfig
) -> PreTrainedModel:
    ## skip vision tower instantiation
    if model_name_or_path is None:
        return None

    vision_tower_arch = None
    if config.resume_path and "radio" not in model_name_or_path:
        assert os.path.exists(
            model_name_or_path
        ), f"Resume vision tower path {model_name_or_path} does not exist!"
        vision_tower_cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        vision_tower_arch = vision_tower_cfg.architectures[0].lower()
    vision_tower_name = (
        vision_tower_arch if vision_tower_arch is not None else model_name_or_path
    )

    if "siglip" in vision_tower_name:
        vision_tower = SiglipVisionTower(model_name_or_path, config)
    else:
        raise ValueError(f"Unknown vision tower: {model_name_or_path}")

    config.mm_hidden_size = vision_tower.config.hidden_size
    return vision_tower

def build_context_provider(
    model_type_or_path: str, config: PretrainedConfig
) -> PreTrainedModel:
    if model_type_or_path is None:
        return None

    ## load from pretrained model
    if config.resume_path:
        assert os.path.exists(
            model_type_or_path
        ), f"Resume context provider path {model_type_or_path} does not exist!"
        return ContextProvider.from_pretrained(
            model_type_or_path, config, torch_dtype=eval(config.model_dtype)
        )
    ## build from scratch
    else:
        mm_projector_cfg = ContextProviderConfig(model_type_or_path)
        mm_projector = ContextProvider(mm_projector_cfg, config).to(
            eval(config.model_dtype)
        )
        return mm_projector
