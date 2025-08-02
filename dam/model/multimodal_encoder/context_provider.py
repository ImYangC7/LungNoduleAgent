# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import torch.nn as nn
import torch
import torch.nn.functional as F
# import deepspeed
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

class ContextProviderConfig(PretrainedConfig):
    model_type = "context_provider"

    def __init__(
        self,
        context_provider_type: str=None, 
        hidden_size=768,
        intermediate_size=3072,
        num_hidden_layers=12,
        num_attention_heads=12,
        num_channels=3,
        num_mask_channels=0,
        image_size=224,
        patch_size=16,
        hidden_act="gelu_pytorch_tanh",
        layer_norm_eps=1e-6,
        attention_dropout=0.0,
        zero_init_output=True,
        residual_dropout=0.0,
        context_image_as_queries=False,
        context_provider_layer_indices=None,
        masked_cross_attn=False,
        crop_position_single_embedding=False,
        trainable_crop_position_embedding=True,
        crop_embedding_mode="add",
        treat_image_as_cimage=False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.context_provider_type = context_provider_type

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.num_mask_channels = num_mask_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act

        self.zero_init_output = zero_init_output
        self.residual_dropout = residual_dropout
        self.context_image_as_queries = context_image_as_queries

        # cross_attn_end_to_all
        # the `num_hidden_layers` should be the same as the one in the vision tower
        self.num_hidden_layers = num_hidden_layers
        self.context_provider_layer_indices = context_provider_layer_indices

        self.masked_cross_attn = masked_cross_attn
        # If enabled, crop_position_embedding (delta to full pos) will be updated during training.
        self.trainable_crop_position_embedding = trainable_crop_position_embedding
        # If enabled, crop_position_embedding (delta to full pos) will be a single embedding for all positions.
        self.crop_position_single_embedding = crop_position_single_embedding
        # add: delta. replace: do not add the original positional embedding
        self.crop_embedding_mode = crop_embedding_mode

        # If True, the input image will be treated as a cimage (with mask as full 1s)
        self.treat_image_as_cimage = treat_image_as_cimage


# Context Provider
from transformers.activations import ACT2FN
from typing import Optional, Tuple

class ContextProviderCrossAttention(nn.Module):
    """Multi-headed cross-attention from 'Attention Is All You Need' paper"""

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention.__init__
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        batch_size, q_len, _ = hidden_states.size()
        batch_size, kv_len, _ = encoder_hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(encoder_hidden_states)
        value_states = self.v_proj(encoder_hidden_states)

        query_states = query_states.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        k_v_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scale

        if attn_weights.size() != (batch_size, self.num_heads, q_len, k_v_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(batch_size, self.num_heads, q_len, k_v_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (batch_size, 1, q_len, k_v_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(batch_size, 1, q_len, k_v_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

            # Visualizations (-inf are shown as white)
            # import matplotlib.pyplot as plt
            # plt.imshow(attention_mask[0, 0, 0].view(27, 27).detach().cpu().numpy())
            # plt.title("Attention mask")
            # plt.colorbar()
            # plt.show()

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        
        # Visualizations: show the attention weights of the first head, with the first query
        # import matplotlib.pyplot as plt
        # plt.imshow(attn_weights[0, 0, 0].view(27, 27).detach().cpu().numpy())
        # plt.title("Attention weights")
        # plt.colorbar()
        # plt.show()

        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(batch_size, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights

class ContextProviderMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


def get_token_mask_bias(mask, patch_size):
    # Note: mask should be (0, 1)
    with torch.no_grad():
        # Add a channel dimension and perform conv
        # mask_tokens_after_conv: (B, 1, H, W), example dimension: [1, 1, 27, 27]
        mask_tokens_after_conv = F.conv2d(
            input=mask[:, None], 
            weight=torch.ones(
                (1, 1, patch_size, patch_size), 
                device=mask.device, dtype=mask.dtype
            ), 
            bias=None, 
            stride=(patch_size, patch_size), 
            padding="valid"
        )
    
        token_mask_bias = torch.zeros_like(mask_tokens_after_conv)
        token_mask_bias.masked_fill_(mask_tokens_after_conv < 1e-5, float("-inf"))
        token_mask_bias = token_mask_bias.flatten(1)
    
    # Flattened dimension: (1, 729)
    return token_mask_bias

def attn_mask_from_cimage_concatenated(cimage_concatenated, patch_size):
    # Use the mask from input image (4th channel)
    mask_normalized = cimage_concatenated[:, 3]
    mask_unnormalized = (mask_normalized + 1) / 2
    # (1, 729)
    token_mask_bias = get_token_mask_bias(mask_unnormalized, patch_size=patch_size)
    
    # attn_mask: (B, 1, Q, KV)
    # print("Token positions:", token_mask.nonzero())

    # Obtain token mask in the bias format: in mask 0, out of mask -inf
    q_kv = token_mask_bias.shape[-1]
    attn_mask_bias = token_mask_bias[:, None, None, :].repeat(1, 1, q_kv, 1)
    
    # Visualizations
    # print(f"token_mask_bias shape: {token_mask_bias.shape}, attn_mask_bias shape: {attn_mask_bias.shape}")
    # import matplotlib.pyplot as plt
    # plt.imshow(attn_mask_bias[0, 0, 0].view(27, 27).detach().cpu().numpy())
    # plt.title("Attention mask (outside)")
    # plt.show()    
    
    return attn_mask_bias

# From SiglipEncoderLayer. We would like to modify this to cross-attention.
class CrossAttnEncoderLayer(nn.Module):
    def __init__(self, config: ContextProviderConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.cross_attn = ContextProviderCrossAttention(config)
        self.residual_dropout = nn.Dropout(config.residual_dropout)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)
        self.mlp = ContextProviderMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps)

        if config.zero_init_output:
            # TODO: alternatively, we could parameterize with an MLP
            # These factors are initialized with 0 (so only residual passes through)
            if config.context_provider_type != "cross_attn_at_the_end":
                self.register_parameter("attn_factor", nn.Parameter(torch.zeros((1,))))
                self.register_parameter("mlp_factor", nn.Parameter(torch.zeros((1,))))
            else:
                # Use scalar tensor for compatibility
                self.register_parameter("attn_factor", nn.Parameter(torch.zeros((1,)).view(())))
                self.register_parameter("mlp_factor", nn.Parameter(torch.zeros((1,)).view(())))
        else:
            self.attn_factor = 1.
            self.mlp_factor = 1.

    # Ignore copy
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`):
                Input to the layer of shape `(batch, seq_len, embed_dim)`.
            attention_mask (`torch.FloatTensor`):
                Attention mask of shape `(batch, 1, q_len, k_v_seq_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.cross_attn(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
        )
        # Dropping the residual: let the model leverage more on the context
        hidden_states = self.residual_dropout(residual) + self.attn_factor * hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + self.mlp_factor * hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class CrossAttnContextProviderEndToAll(nn.Module):
    def __init__(self, config: ContextProviderConfig):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttnEncoderLayer(config) for i in enumerate(range(config.num_hidden_layers)) if config.context_provider_layer_indices is None or i in config.context_provider_layer_indices
        ])
        self.patch_size = config.patch_size
        self.masked_cross_attn = config.masked_cross_attn

    def forward(self, context_image_features, cimage_concatenated, vision_tower):
        # Use the mask from input image (4th channel)
        if self.masked_cross_attn:
            attn_mask = attn_mask_from_cimage_concatenated(cimage_concatenated, patch_size=self.patch_size)
        else:
            attn_mask = None
        
        detail_raw_image = cimage_concatenated[:, 4:, ...]
        # NOTE: when using context image as queries, the context image was swapped with the detail image before passing into the context provider
        outputs = vision_tower(detail_raw_image, context_provider_layers=self.layers, contexts=context_image_features, cross_attention_mask=attn_mask)

        return outputs

class ContextProvider(PreTrainedModel):
    config_class = ContextProviderConfig

    def __init__(
        self, context_provider_cfg: ContextProviderConfig, config: PretrainedConfig
    ):
        super().__init__(context_provider_cfg)

        self.context_image_as_queries = context_provider_cfg.context_image_as_queries
        self.context_provider_type = context_provider_type = context_provider_cfg.context_provider_type

        self.treat_image_as_cimage = context_provider_cfg.treat_image_as_cimage
        
        if self.context_image_as_queries:
            assert not context_provider_cfg.masked_cross_attn, "Masked cross-attention not implemented when using context image as queries."
            assert "concat" not in context_provider_type, "Concat not implemented when using context image as queries."
        
        if context_provider_type == "cross_attn_end_to_all":
            # Information flow: end of context features -> all detail features
            self.context_provider_module = CrossAttnContextProviderEndToAll(context_provider_cfg)
        else:
            raise ValueError(f"Unknown context provider type: {context_provider_type}")

    def forward(self, cimage_full_features=None, cimage_crop_features=None, cimage_concatenated=None, vision_tower=None):
        if self.context_provider_type == "cross_attn_end_to_all":
            assert cimage_full_features.shape[0] == cimage_concatenated.shape[0], f"shape mismatches: {cimage_full_features.shape[0]} != {cimage_concatenated.shape[0]}"
            return self.context_provider_module(context_image_features=cimage_full_features, cimage_concatenated=cimage_concatenated, vision_tower=vision_tower)
        else:
            raise ValueError(f"Unknown context provider type: {context_provider_type}")

AutoConfig.register("context_provider", ContextProviderConfig)
AutoModel.register(ContextProviderConfig, ContextProvider)
