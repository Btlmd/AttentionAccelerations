from torch import nn
import torch
import math
from dgl.nn.functional import edge_softmax
import dgl.function as fn
from .diffuser.diffuser_utils import *
from .diffuser.utils import *
import dgl
import logging
import time

class DiffuserSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_head != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_head})"
            )
        self.num_heads = config.num_head
        self.head_dim = int(config.hidden_size / config.num_head)
        self.embed_dim = config.hidden_size

        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)

        self.dropout = config.attention_probs_dropout_prob

        attention_window = config.attention_window
        assert (
            attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
            attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self._create_adj_mat()

        self.cached_g = None
        self.cached_info = None

    def _create_adj_mat(self):
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )
        max_len = 4096 # not the input sequence max len
        n_blocks = max_len//(attention_window//2)-1
        adj = np.zeros([max_len, max_len])
        
        # add local window att (overlap)
        for i in range(n_blocks):
            start = i*attention_window//2
            end = start+attention_window
            if end > max_len:
                end = max_len
            adj[start:end, start:end] = 1

        # add random att    
        np.random.seed(0)
        num_random = max_len*self.config.num_rand
        
        idx = np.random.choice(range(max_len*max_len), num_random ,replace=False)
        idx_x = idx %  max_len
        idx_y = idx // max_len
        adj[idx_x,idx_y] = 1

        # add global att    
        num_global = self.config.num_glob
        idx = np.random.choice(range(attention_window,max_len), num_global ,replace=False)
        adj[idx,:] = 1
        adj[:,idx] = 1

        possible_seq_len = np.arange(attention_window, max_len+attention_window, attention_window)
        self.src_dst = {k: np.nonzero(adj[:k, :k]) for k in possible_seq_len}

    def _pad_to_window_size(self,inputs):
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )
        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = inputs["input_ids"].shape if inputs["input_ids"] is not None else inputs["attention_mask"].shape
        batch_size, seq_len = input_shape[:2]
        padding_len = (attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:
            logging.debug(
                f"Input ids are automatically padded from {seq_len} to {seq_len + padding_len} to be a multiple of "
                f"`config.attention_window`: {attention_window}"
            )
            if inputs["input_ids"] is not None:
                inputs["input_ids"] = nn.functional.pad(inputs["input_ids"], (0, padding_len), value=self.config.pad_token_id)
            inputs["attention_mask"] = nn.functional.pad(
                inputs["attention_mask"], (0, padding_len), value=False
            )  # no attention on the padding tokens
        return inputs

    def _from_adj_to_batched_graphs(self, B, seq_len):
        if self.cached_g is not None and self.cached_info is not None and self.cached_info == (B, seq_len):
            return self.cached_g
        print("create new graph", B, seq_len)
        g_list = []
        for i in range(B):
            src,dst =self.src_dst[seq_len]
            g = dgl.graph((src, dst))
            g_list.append(g)
        batched_g = dgl.batch(g_list).to('cuda')
        self.cached_g = batched_g
        self.cached_info = (B, seq_len)
        return batched_g
    

    def forward(
        self,
        hidden_states,
        attention_mask=None,
    ):
        # start = time.time()
        g = self._from_adj_to_batched_graphs(hidden_states.size(0), hidden_states.size(1))
        # print("Elapsed time: ", round(time.time()-start, 2))

        if (attention_mask >= 0).all():
            attention_mask[attention_mask == 0] = -10000.0
            attention_mask[attention_mask == 1] = 0.0

        hidden_states = hidden_states.transpose(0, 1) #(N,B,HD)
        # attention_mask (B,N)
        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)   # (N,B,HD)

        seq_len, batch_size, embed_dim = hidden_states.size()
        assert (
            embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)

        query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1) # (B,N,H,D)
        key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1) # (B,N,H,D)
        value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads, self.head_dim).transpose(0, 1) #B,N,H,D
        
        bool_mask = (attention_mask>=0 )
        g = g.local_var()
        g.ndata["mask"] = bool_mask.reshape(-1).unsqueeze(-1)        #BN,1
        g.ndata['q'] =  query_vectors.reshape(-1, self.num_heads, self.head_dim) #BN,H,D
        g.ndata['k'] =  key_vectors.reshape(-1, self.num_heads, self.head_dim) #BN,H,D~
        g.ndata['v'] =  value_vectors.reshape(-1, self.num_heads, self.head_dim) #BN,H,D
        
        g.apply_edges(fn.u_dot_v('k', 'q', 'score'))   #score: [E,H,1]
        g.apply_edges(mask_attention_score)   #kq
        e = g.edata.pop('score') 
        g.edata['score'] = edge_softmax(g, e)
        g.edata['score']= nn.functional.dropout(g.edata['score'], p=self.dropout, training=self.training)
        
        g.ndata["h"] = g.ndata["v"]
        alpha = 0.1
        for _ in range(5):
            g.update_all(fn.u_mul_e('h', 'score', 'm'), fn.sum('m', 'h'))
            g.apply_nodes(lambda nodes: {'h' : (1.0 - alpha) * nodes.data['h'] + alpha * nodes.data['v']})
            g.ndata['h']= nn.functional.dropout(g.ndata['h'], p=self.dropout, training=self.training)

        attn_output = g.ndata['h'] #BN,H,D
        attn_output = attn_output.reshape(batch_size, seq_len,  self.num_heads, self.head_dim) # B,N,H,D        
        assert attn_output.size() == (batch_size, seq_len, self.num_heads, self.head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size, embed_dim).contiguous()
        outputs = attn_output.transpose(0, 1) # Seq,B,D

        # print("Elapsed time end: ", round(time.time()-start, 2))

        return outputs # B, Seq, HD


