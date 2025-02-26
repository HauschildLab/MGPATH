# coding=utf-8
"""
@author : Tien Nguyen
@date   : 2024-Nov-09
@update : 2025-Feb-25
"""
import torch
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask

from models import Adapter

class PLIPTextEncoder(torch.nn.Module):
    def __init__(
        self,
        projector
        #dual_mlp=False
    ) -> None:
        super().__init__()
        print("use PLIP Text Encoder")
        self.transformer = projector.text_model.text_model.encoder
        self.final_layer_norm = projector.text_model.text_model.final_layer_norm
        self.eos_token_id = projector.text_model.text_model.eos_token_id
        self.proj = projector.TextMLP
        #self.dual_mlp = dual_mlp
        #if dual_mlp:
        #    print("=======Using dual MLP text==========")
        #    self.proj_new = Adapter(image=False, hidden=512)

    def forward(
        self,
        prompts,
        attention_mask,
        tokenized_prompts
    ) -> torch.Tensor:
        input_shape = tokenized_prompts.size()
        causal_attention_mask = _create_4d_causal_attention_mask(
                        input_shape, prompts.dtype, device=prompts.device
        )
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask,\
                                                            prompts.dtype)
        encoder_outputs = self.transformer(
            inputs_embeds=prompts.to(prompts.device),
            attention_mask=attention_mask.to(prompts.device),
            causal_attention_mask=causal_attention_mask,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0],\
                                        device=last_hidden_state.device),
                tokenized_prompts.to(dtype=torch.int,\
                        device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0],\
                                        device=last_hidden_state.device),
                (tokenized_prompts.to(dtype=torch.int,\
                    device=last_hidden_state.device) == self.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]

        projected = self.proj(pooled_output)
        #if self.dual_mlp:
        #    projected_new = self.proj_new(pooled_output)
        #else:
            #projected_new = 0
        #return projected, projected_new
        return projected, 0

