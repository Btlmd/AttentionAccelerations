import torch
import torch.nn.functional as F
import numpy as np

from torch import Tensor
from typing import Optional
from torch import nn

class CosformerAttention(nn.Module):
    """
    cosformer attention in "cosFormer: Rethinking Softmax In Attention"
    https://arxiv.org/abs/2202.08791
    """
    def __init__(self,config):

        super().__init__()

        self.act_fun = self.get_act_fun("relu")


    def get_index(self, seq_len):
        index = np.pi / 2 * torch.arange(1, seq_len + 1).reshape(1, -1, 1)

        return nn.Parameter(index, requires_grad=False)

    def get_act_fun(self, act_fun):
        if act_fun == "relu":
            return F.relu
        elif act_fun == "elu":
            return 1 + F.elu

    def forward(
        self,
        q: Tensor,
        k: Optional[Tensor] = None,
        v: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
        eps: Optional[float] = 1e-6,
    ):
           
        bsz, head_num, tgt_len, dim = q.shape
        src_len = k.shape[2]
   
        q = q.reshape(-1, q.shape[-2], q.shape[-1])
        k = k.reshape(-1, k.shape[-2], k.shape[-1])
        v = v.reshape(-1, v.shape[-2], v.shape[-1])


        # activation
        q = self.act_fun(q)
        k = self.act_fun(k)

        
        # cos transform
        m = max(src_len, tgt_len)
        # get index and send to cuda
        weight_index = self.get_index(m).to(q)
        # (N * h, L, 2 * d)
        q_ = torch.cat([q * torch.sin(weight_index[:, :tgt_len, :] / m), q * torch.cos(weight_index[:, :tgt_len, :] / m)], dim=-1)
        # (N * h, S, 2 * d)
        k_ = torch.cat([k * torch.sin(weight_index[:, :src_len, :] / m), k * torch.cos(weight_index[:, :src_len, :] / m)], dim=-1)

    
        ## Need to improve speed!
        # (N * h, L, 2 * d) (N * h, L, d) -> (N * h, L, h, 2 * d, d)
        kv_ = torch.einsum("nld,nlm->nldm", k_, v)
        # (N * h, L, 2 * d, d) -> (N * h, L, 2 * d, d)
        kv_cum = torch.cumsum(kv_, dim=1)
        # (N * h, L, 2 * d) (N * h, L, 2 * d, d) -> (N * h, L, d)
        qkv = torch.einsum("nld,nldm->nlm", q_, kv_cum)
        # (N * h, L, 2 * d) -> (N * h, L, 2 * d)
        k_cum = torch.cumsum(k_, dim=1)
        # (N * h, L, 2 * d) (N * h, L, 2 * d) -> (N * h, L)
        denom = torch.clamp_min(torch.einsum("nlm,nlm->nl", q_, k_cum), eps)
        # (N * h, L, d) (N * h, L, 1) -> (N * h, L, d)
        attn_output = qkv / denom.unsqueeze(-1)

        attn_output = attn_output.reshape(bsz, head_num, attn_output.shape[-2], attn_output.shape[-1])

        return attn_output
