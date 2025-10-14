# coding=utf-8
"""
@desc:
    - The original implementation by https://github.com/Jiangbo-Shi/ViLa-MIL
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import logging

import math
from os.path import join as pjoin

logger = logging.getLogger(__name__)

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch_geometric.nn import GCN, GATConv, GraphConv
from transformers.modeling_attn_mask_utils import _create_4d_causal_attention_mask, _prepare_4d_attention_mask
from transformers import CLIPModel, CLIPProcessor
_tokenizer = _Tokenizer()


class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.permute(1, 0, 2)
        x = self.ln_final(x).type(self.dtype)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

class MLP(nn.Module):
    def __init__(self, image = True, hidden=768, out=1024):
        super(MLP, self).__init__()
        if image:
            self.linear1 = nn.Linear(1536, out)
        else:
            self.linear1 = nn.Linear(hidden, out)
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(out, out)

    def forward(self, x):
        y = self.linear1(x)
        y = self.gelu(y)
        out = self.linear2(y)
        return out

class ProjectorPLIP(nn.Module):
    def __init__(self):
        super(ProjectorPLIP, self).__init__()
        print("use PLIP Projector")
        self.ImageMLP = MLP(image = True, hidden=512)
        self.TextMLP = MLP(image = False, hidden=512)
        self.temperature = torch.nn.Parameter(torch.tensor([np.log(1/0.02)]), requires_grad=True)

        self.text_model = CLIPModel.from_pretrained("vinid/plip")

class ProjectorPLIP_only(nn.Module):
    def __init__(self):
        super(ProjectorPLIP_only, self).__init__()
        print("use PLIP_only Projector")
        self.text_model = CLIPModel.from_pretrained("vinid/plip")
        self.TextMLP = self.text_model.text_projection
        self.temperature = self.text_model.logit_scale

class PLIPTextEncoder(nn.Module):
    def __init__(self, projector):
        super().__init__()
        print("use PLIP Text Encoder")
        self.transformer = projector.text_model.text_model.encoder
        self.final_layer_norm = projector.text_model.text_model.final_layer_norm
        self.eos_token_id = projector.text_model.text_model.eos_token_id
        self.proj = projector.TextMLP

    def forward(self, prompts, attention_mask, tokenized_prompts):
        input_shape = tokenized_prompts.size()
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, prompts.dtype, device=prompts.device
        )
        if attention_mask is not None:
            attention_mask = _prepare_4d_attention_mask(attention_mask, prompts.dtype)
        encoder_outputs = self.transformer(
            inputs_embeds=prompts.to(prompts.device),
            attention_mask=attention_mask.to(prompts.device),
            causal_attention_mask=causal_attention_mask,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                tokenized_prompts.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                (tokenized_prompts.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]

        projected = self.proj(pooled_output)
        return projected

class PromptLearner(nn.Module):
    def __init__(self, classnames, clip_model, use_gigapath_backbone=False, use_plip_backbone=False):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = 16

        if use_gigapath_backbone or use_plip_backbone:
            dtype = clip_model.text_model.text_model.embeddings.token_embedding.weight.dtype
            ctx_dim = clip_model.text_model.text_model.embeddings.token_embedding.weight.shape[1]

        self.N = 4 #N = 4

        ctx_vectors = torch.empty(self.N, n_ctx, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors, std=0.02)

        self.ctx = nn.Parameter(ctx_vectors)
        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [name for name in classnames]

        plip_tokenizer = CLIPProcessor.from_pretrained("vinid/plip")
        tokenized_prompts = plip_tokenizer(
            prompts, return_tensors="pt",
            max_length=77,
            padding="max_length",
            truncation=True
        )
        tokenized_prompts['input_ids'] = tokenized_prompts['input_ids'].repeat(self.N,1)
        tokenized_prompts['attention_mask'] = tokenized_prompts['attention_mask'].repeat(self.N,1)
        with torch.no_grad():
            embedding = clip_model.text_model.text_model.embeddings(
                input_ids=tokenized_prompts['input_ids']
            ).type(dtype)

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


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def group_patches(patch_embeddings, positions, grid_size):
    """
    @desc:
        - groups patch embeddings based on spatial positions.
        - the variable grouped_embeddings: is the list of list.
            Each element is the list of embedding. Therefore, each element
                                                represents a group of embeddings
    @params:
        - patch_embeddings (torch.Tensor): Tensor of shape (N, D).
        - positions (torch.Tensor): Tensor of shape (N, 2)
                                                containing (x, y) coordinates.
        - grid_size (int): Number of groups along one axis
                                            (e.g., grid_size=4 for a 4x4 grid).
    @returns:
        - grouped_embeddings (List[torch.Tensor]): List of tensors,
                                    each containing embeddings for one group.
    """
    N, D = patch_embeddings.shape
    groups = [[] for _ in range(grid_size * grid_size)]

    x_coords = positions[:, 0] / positions[:, 0].max()
    y_coords = positions[:, 1] / positions[:, 1].max()

    epsilon = 1e-6
    x_coords = x_coords - epsilon
    y_coords = y_coords - epsilon

    x_bins = torch.floor(x_coords * grid_size).clamp(min=0, max=grid_size - 1).long()
    y_bins = torch.floor(y_coords * grid_size).clamp(min=0, max=grid_size - 1).long()

    group_indices = y_bins * grid_size + x_bins

    for i in range(N):
        group_idx = group_indices[i]
        groups[group_idx].append(patch_embeddings[i])

    grouped_embeddings = []
    for group in groups:
        if group:
            grouped_embeddings.append(torch.stack(group))
        else:
            grouped_embeddings.append(torch.empty(0, D))

    return grouped_embeddings

def GNNmodel(typeGNN, config):
    if typeGNN == "gcn":
        model = GCN(
            in_channels=config.input_size,
            hidden_channels=config.input_size,
            num_layers=1,
            jk='cat'
        )
    elif typeGNN == "gat_conv":
        model = GATConv(
            in_channels=config.input_size,
            out_channels=config.input_size,
        )
    else:
        raise Exception(f" Type {typeGNN} NOT IMPLEMENTED.")
    return model

class MGPATH(nn.Module):
    def __init__(
        self,
        config,
        num_classes=3
    ):
        super(MGPATH, self).__init__()
        self.loss_ce = nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.L = config.input_size
        self.D = config.hidden_size
        self.K = 1
        self.N = 4
        self.eps = 0.1
        self.max_iter = 100

        self.use_gigapath_backbone = config.use_gigapath_backbone
        self.use_plip_backbone = config.use_plip_backbone
        self.ratio_graph = config.ratio_graph

        if config.use_plip_backbone:
            clip_model = ProjectorPLIP_only()
            self.text_encoder = PLIPTextEncoder(clip_model)
        elif config.use_gigapath_backbone:
            clip_model = ProjectorPLIP()
            checkpoint_dict = torch.load("checkpoints/clip_mlp_weights_dual_tokenSave_plip_best.pth")
            clip_model.load_state_dict(checkpoint_dict['model_state_dict'])
            self.text_encoder = PLIPTextEncoder(clip_model)
            self.ImageMLP = clip_model.ImageMLP

        if config.freeze_text_encoder:
            print('\nFreeze text encoder...')
            if config.use_plip_backbone or config.use_gigapath_backbone:
                for name, param in self.text_encoder.named_parameters(): 
                    if "mlp_ratio" not in name:
                        param.requires_grad = False
                for param in self.text_encoder.proj.parameters():
                    param.requires_grad = True

        self.prompt_learner = PromptLearner(
            config.text_prompt, clip_model.float(), config.use_gigapath_backbone, config.use_plip_backbone
        )

        if config.use_plip_backbone or config.use_gigapath_backbone:
            self.logit_scale = clip_model.temperature
        else:
            self.logit_scale = clip_model.logit_scale

        self.norm = nn.LayerNorm(config.input_size)

        self.graph = GNNmodel(config.typeGNN, config)

        self.learnable_image_center = nn.Parameter(torch.Tensor(*[64, 1, config.input_size]))
        trunc_normal_(self.learnable_image_center, std=.02)

    def graph_adapter(self, node_features, edge_index):
        return self.graph(node_features, edge_index)


    def standard_attention(
        self,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor
    ):
         d_k = self.learnable_image_center.size(-1)
         attn_scores = torch.matmul(self.learnable_image_center, k_tensor.transpose(-2, -1)) / math.sqrt(d_k)
         attention_weights = F.softmax(attn_scores, dim=-1)
         attn_patches = torch.matmul(attention_weights, v_tensor)
         return attn_patches


    def aggregator(
        self,
        input_x: torch.Tensor,
        M_graph: torch.Tensor,
    ) -> torch.Tensor:
        """
        @desc:
            - `input_x` has the shape of [N, 1024]
                                            where N is the number of patches
            - `input_x` can be:
                1. input_x = x_s.float()
                2. input_x = x_l.float()
        """
        attn_patches1 = self.standard_attention(input_x, input_x)
        attn_patches2 = self.standard_attention(M_graph, M_graph)
        p=self.ratio_graph
        attn_patches = (1 - p) * attn_patches1 + p*attn_patches2 + self.learnable_image_center
        attn_patches = self.norm(attn_patches)
        return attn_patches


    def forward(
        self,
        x_s,
        node_s,
        edge_s,
        x_l,
        node_l,
        edge_l,
        label
    ):
        self.n_cls = self.num_classes * 2
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts

        text_features = self.text_encoder(
            prompts, tokenized_prompts['attention_mask'], tokenized_prompts['input_ids']
        )

        text_features =  text_features.contiguous().view(self.N, self.n_cls, self.L)

        M_low = x_s.float()
        M_low = M_low.squeeze(0)

        M_low = self.ImageMLP(M_low)
        node_s = self.ImageMLP(node_s.squeeze(0).float())

        M_graph_low = self.graph_adapter(node_s, edge_s.squeeze(0))
        compents_attn_patches = self.aggregator(M_low, M_graph_low)
  
        M_high = x_l.float()
        M_high = M_high.squeeze(0)

        M_high = self.ImageMLP(M_high)
        node_l = self.ImageMLP(node_l.squeeze(0).float())

        M_graph_high = self.graph_adapter(node_l, edge_l.squeeze(0))
        compents_high_attn_patches = self.aggregator(M_high, M_graph_high)

        image_features_low_norm = torch.nn.functional.normalize(compents_attn_patches, p=2, dim=2)
        image_features_high_norm = torch.nn.functional.normalize(compents_high_attn_patches, p=2, dim=2)
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=2)

        text_features_low = text_features[:, :self.num_classes, :]
        text_features_high = text_features[:, self.num_classes:, :]

        logits_ot_low = do_ot(
            image_features=image_features_low_norm,
            text_features=text_features_low,
            logit_scale_fct=self.logit_scale
        )
        logits_ot_high = do_ot(
            image_features=image_features_high_norm,
            text_features=text_features_high,
            logit_scale_fct=self.logit_scale
        )

        logits = logits_ot_low + logits_ot_high

        loss = self.loss_ce(logits, label)
        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.topk(Y_prob, 1, dim = 1)[1]

        return Y_prob, Y_hat, loss


def do_ot(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    logit_scale_fct
):
    """
    @desc:
        - do OPTIMAL TRANSPORT (OT) for low or high resolution
    @params:
        - image_features has the shape of [16, 1, 1024]
            1 is the batch size (for 1 slide or 1 WSI)
        - text_features has the shape of [4, 2, 1024]
    """
    b = 1
    N = 4
    eps = 0.1
    n_cls = 2
    M = image_features.shape[0]

    sim = torch.einsum('mbd,ncd->mnbc', image_features, text_features).contiguous()
    sim = sim.view(M, N, b*n_cls)
    sim = sim.permute(2,0,1)
    wdist = 1.0 - sim
    xx=torch.zeros(b*n_cls, M, dtype=sim.dtype, device=sim.device).fill_(1. / M)
    yy=torch.zeros(b*n_cls, N, dtype=sim.dtype, device=sim.device).fill_(1. / N)
    with torch.no_grad():
        KK = torch.exp(-wdist / eps)
        T = Sinkhorn(KK,xx,yy)
        if torch.isnan(T).any():
            return None

    sim_op = torch.sum(T * sim, dim=(1, 2))
    sim_op = sim_op.contiguous().view(b, n_cls)

    logit_scale = logit_scale_fct.exp()
    logits2 = logit_scale * sim_op
    return logits2


def Sinkhorn(K, u, v):
    max_iter = 100
    r = torch.ones_like(u)
    c = torch.ones_like(v)
    thresh = 1e-2
    for i in range(max_iter):
        r0 = r
        r = u / torch.matmul(K, c.unsqueeze(-1)).squeeze(-1)
        c = v / torch.matmul(K.permute(0, 2, 1).contiguous(), r.unsqueeze(-1)).squeeze(-1)
        err = (r - r0).abs().mean()
        if err.item() < thresh:
            break

    T = torch.matmul(r.unsqueeze(-1), c.unsqueeze(-2)) * K

    return T
