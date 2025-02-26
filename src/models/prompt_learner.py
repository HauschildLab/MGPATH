# coding=utf-8
"""
@author : Tien Nguyen
@date   : 2024-Nov-09
@update : 2025-Feb-25
"""
import torch

import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

from transformers import CLIPProcessor

_tokenizer = _Tokenizer()

class PromptLearner(torch.nn.Module):
    def __init__(
        self,
        classnames,
        clip_model,
    ) -> None:
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16
        ctx_init = ""

        dtype = clip_model.text_model.text_model.embeddings.token_embedding.weight.dtype
        ctx_dim = clip_model.text_model.text_model.embeddings.token_embedding.weight.shape[1]
        self.N = 4 #N = 4

        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if False:
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("================generic context vector===============")
                ctx_vectors = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)
            torch.nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        self.ctx = torch.nn.Parameter(ctx_vectors)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [name for name in classnames]

        plip_tokenizer = CLIPProcessor.from_pretrained("vinid/plip")
        tokenized_prompts = plip_tokenizer(prompts, return_tensors="pt",\
                                    max_length=77, padding="max_length",\
                                                truncation=True) # (n_cls, 77)
        tokenized_prompts['input_ids'] = \
            tokenized_prompts['input_ids'].repeat(self.N,1) # (N*n_cls, 77)
        tokenized_prompts['attention_mask'] = \
                            tokenized_prompts['attention_mask'].repeat(\
                                                    self.N,1) # (N*n_cls, 77)
        with torch.no_grad():
            embedding = clip_model.text_model.text_model.embeddings(\
                        input_ids=tokenized_prompts['input_ids']).type(\
                                                    dtype) # (N*n_cls, 77, 512)

        self.register_buffer("token_prefix", embedding[:, :1, :])
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = "end"

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 3:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1,-1)

        ctx = ctx.permute(1, 0, 2, 3)
        ctx = ctx.contiguous().view(self.N*self.n_cls,self.n_ctx,ctx.shape[3])

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat([prefix,ctx,suffix,],dim=1,)
        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        ctx_i_half1,
                        class_i,
                        ctx_i_half2,
                        suffix_i,
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,
                        class_i,
                        ctx_i,    # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        else:
            raise ValueError
        return prompts
