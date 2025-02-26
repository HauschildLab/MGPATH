# coding=utf-8
"""
@author : Tien Nguyen
@date   : 2024-Nov-09
@update : 2025-Feb-25
"""
import torch


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
    d = image_features.shape[-1]

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
