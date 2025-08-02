#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

# This file is modified from https://github.com/haotian-liu/LLaVA/ and https://github.com/NVlabs/VILA/

import torch
from transformers import (
    AutoConfig,
    BitsAndBytesConfig,
    PretrainedConfig,
)

from .language_model.llava_llama import LlavaLlamaModel
# TODO: we may move LlavaConfig to configuration_llava.py
# from model.configuration_llava import LlavaConfig


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch
    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def load_pretrained_model(
    model_path,
    model_name,
    model_base=None,
    load_8bit=False,
    load_4bit=False,
    device_map="auto",
    device="cuda",
    **kwargs,
):
    kwargs = {"device_map": device_map, **kwargs}

    if device != "cuda":
        kwargs["device_map"] = {"": device}

    if load_8bit:
        kwargs["load_in_8bit"] = True
    elif load_4bit:
        kwargs["load_in_4bit"] = True
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    config = AutoConfig.from_pretrained(model_path)
    config.resume_path = model_path
    prepare_config_for_eval(config, kwargs)

    model = LlavaLlamaModel(
        config=config,
        low_cpu_mem_usage=True,
        **kwargs
    )
    tokenizer = model.tokenizer

    model.eval()
    
    # mm_use_im_start_end = getattr(
    #     model.config, "mm_use_im_start_end", False)
    # mm_use_im_patch_token = getattr(
    #     model.config, "mm_use_im_patch_token", True)
    # if mm_use_im_patch_token:
    #     tokenizer.add_tokens(
    #         [DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    # if mm_use_im_start_end:
    #     tokenizer.add_tokens(
    #         [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
    #     )

    model.resize_token_embeddings(len(tokenizer))
    vision_tower = model.get_vision_tower()
    vision_tower.to(device=device, dtype=torch.float16)
    mm_projector = model.get_mm_projector()
    mm_projector.to(device=device, dtype=torch.float16)
    context_provider = model.get_context_provider()
    if context_provider is not None:
        context_provider.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor

    if hasattr(model.llm.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len


def parse_model_name_or_path(config: PretrainedConfig, model_name="llm", suffix="_cfg"):
    target_model = f"{model_name}{suffix}"
    target_cfg = getattr(config, target_model, None)

    if isinstance(target_cfg, str):
        return target_cfg
    elif isinstance(target_cfg, dict):
        return target_cfg["architectures"][0]
    else:
        raise ValueError(f"Invalid {target_model} configuration!")


def prepare_config_for_eval(config: PretrainedConfig, kwargs: dict):
    try:
        # compatible with deprecated config convention
        if getattr(config, "vision_tower_cfg", None) is None:
            config.vision_tower_cfg = config.mm_vision_tower
    except AttributeError:
        raise ValueError(
            f"Invalid configuration! Cannot find vision_tower in config:\n{config}")

    config.model_dtype = kwargs.pop("torch_dtype").__str__()
    # siglip does not support device_map = "auto"
    vision_tower_name = parse_model_name_or_path(config, "vision_tower")
    if "siglip" in vision_tower_name.lower():
        kwargs["device_map"] = "cuda"
