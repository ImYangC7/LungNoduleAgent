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

import torch # noqa
from .vision_encoder import VisionTower

from transformers import AutoConfig, PretrainedConfig, AutoModel
from .siglip import (
    SiglipVisionConfig,
    SiglipVisionModel,
    SiglipImageProcessor,
)


class SiglipVisionTower(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig, state_dict=None):
        super().__init__(model_name_or_path, config)
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path)
        self.vision_tower = SiglipVisionModel.from_pretrained(
            # TODO(ligeng): why pass config here leading to errors?
            model_name_or_path, torch_dtype=eval(config.model_dtype), state_dict=state_dict
        )
        self.is_loaded = True


AutoConfig.register("siglip_vision_model", SiglipVisionConfig, exist_ok=True)
AutoModel.register(SiglipVisionConfig, SiglipVisionModel, exist_ok=True)

