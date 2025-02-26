# coding=utf-8
"""
@author : Tien Nguyen, Duy Nguyen, Nghiem Diep, Trung Nguyen
@modify : Tien Nguyen
@date   : 2024-Nov-09
@update : 2025-Feb-25
"""
import math

import torch
from torch.nn import functional as F
from torch_geometric.nn import GATConv

import utils
from models import PLIPProjector
from models import PLIPTextEncoder
from models import PromptLearner


def _no_grad_trunc_normal_(
    tensor,
    mean,
    std,
    a,
    b
):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor

def trunc_normal_(
    tensor,
    mean=0.,
    std=1.,
    a=-2.,
    b=2.
):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

def create_gnn_model(
    typeGNN,
    config
):
    model = None
    if typeGNN == "gat_conv":
        model = GATConv(
            in_channels=config['input_size'],     
            out_channels=config['input_size'],  
        )
    else:
        raise Exception("Wrong type GNN.")
    return model

class MILModel(torch.nn.Module):
    def __init__(
        self,
        config,
        num_classes=3
    ) -> None:
        super(MILModel, self).__init__()
        self.loss_ce = torch.nn.CrossEntropyLoss()
        self.num_classes = num_classes
        self.L = config['input_size']
        self.K = 1
        self.N = 4
        self.eps = 0.1
        self.max_iter = 100

        self.ratio_graph = config['ratio_graph']

        clip_model = PLIPProjector()
        checkpoint_dict = torch.load(config['alignment'])
        clip_model.load_state_dict(checkpoint_dict['model_state_dict'])

        self.text_encoder = PLIPTextEncoder(clip_model)
        self.ImageMLP = clip_model.ImageMLP

        for name, param in self.text_encoder.named_parameters():
            param.requires_grad = False

        for param in self.text_encoder.proj.parameters():
            param.requires_grad = True


        self.prompt_learner = PromptLearner(config['text_prompt'],\
                                                            clip_model.float())
        self.logit_scale = clip_model.temperature

        self.norm = torch.nn.LayerNorm(config['input_size'])

        self.graph = create_gnn_model(config['typeGNN'], config)

        self.learnable_image_center = torch.nn.Parameter(\
                                    torch.Tensor(*[64, 1, config['input_size']]))
        trunc_normal_(self.learnable_image_center, std=.02)

    def graph_adapter(self, node_features, edge_index):
        return self.graph(node_features, edge_index)

    def standard_attention(
        self,
        q_tensor: torch.Tensor,
        k_tensor: torch.Tensor,
        v_tensor: torch.Tensor
    ):
        d_k = self.learnable_image_center.size(-1)
        attn_scores = torch.matmul(self.learnable_image_center,\
                                k_tensor.transpose(-2, -1)) / math.sqrt(d_k)
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
        attn_patches1 = self.standard_attention(self.learnable_image_center,\
                                                            input_x, input_x)
        attn_patches2 = self.standard_attention(self.learnable_image_center,\
                                                            M_graph, M_graph)

        p=self.ratio_graph
        attn_patches = (1-p)*attn_patches1 + p*attn_patches2 + self.learnable_image_center #(64, 1, 1024)
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
        n_ctx = 16
        self.d = self.L
        self.n_cls = 4 #2 classes for low resolution and 2 classes for high resolution
        prompts = self.prompt_learner()
        tokenized_prompts = self.prompt_learner.tokenized_prompts
        text_features, text_features_new = self.text_encoder(prompts,\
                                        tokenized_prompts['attention_mask'],\
                                                tokenized_prompts['input_ids'])

        text_features =  text_features.contiguous().view(self.N,\
                                                            self.n_cls, self.d)

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

        M_graph_high = self.graph_adapter(node_l, edge_l.squeeze(0))
        compents_high_attn_patches = self.aggregator(M_high, M_graph_high)

        image_features_low_norm = torch.nn.functional.normalize(\
                                            compents_attn_patches, p=2, dim=2)
        image_features_high_norm = torch.nn.functional.normalize(\
                                        compents_high_attn_patches, p=2, dim=2)
        text_features = torch.nn.functional.normalize(text_features, p=2, dim=2)

        text_features_low = text_features[:, :self.num_classes, :]
        text_features_high = text_features[:, self.num_classes:, :]

        logits_ot_low = utils.do_ot(image_features=image_features_low_norm,\
                                            text_features=text_features_low,\
                                            logit_scale_fct=self.logit_scale)
        logits_ot_high = utils.do_ot(image_features=image_features_high_norm,\
                                            text_features=text_features_high,\
                                            logit_scale_fct=self.logit_scale)

        logits = logits_ot_low + logits_ot_high

        loss = self.loss_ce(logits, label)
        Y_prob = F.softmax(logits, dim = 1)
        Y_hat = torch.topk(Y_prob, 1, dim = 1)[1]

        return Y_prob, Y_hat, loss
