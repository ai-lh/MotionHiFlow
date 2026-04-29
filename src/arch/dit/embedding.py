import torch
import torch.nn as nn
import numpy as np
import math
from einops import rearrange
from torch import Tensor

class TimestepEmbedding(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.scale = nn.Parameter(torch.ones(1))
        self.frequency_embedding_size = frequency_embedding_size

    @property
    def device(self):
        return next(self.parameters()).device

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        # t: [batch_size] or ()
        if isinstance(t, int | float):
            t = torch.tensor([t]).to(self.device)
        if t.ndim == 0:
            t = t.unsqueeze(-1)
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq * self.scale)
        return t_emb

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) #[1, max_len, d_model]

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)



class LearnableEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(LearnableEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(max_len, d_model)))

    def forward(self, x):
        """
        x: [..., seq, dim]
        """
        return self.dropout(self.embedding[:x.shape[-2], :]) + x


class JointPosEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(JointPosEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.embedding22 = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(22, d_model)))
        self.embedding21 = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(21, d_model)))
        self.embedding12 = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(12, d_model)))
        self.embedding7 = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(7, d_model)))
        self.embedding5 = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(5, d_model)))
        self.embedding2 = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(2, d_model)))
        self.direc = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(1, d_model)))

        self.map_right = {
            22: [1, 4, 7, 10, 13, 16, 18, 20],
            21: [8, 9, 10, 16, 17, 18, 19, 20],
            12: [1, 2, 7, 8],
            7: [1, 4],
            5: [1, 3],
            2: [],
        }
        self.map_left = {
            22: [2, 5, 8, 11, 14, 17, 19, 21],
            21: [5, 6, 7, 11, 12, 13, 14, 15],
            12: [3, 4, 9, 10],
            7: [2, 5],
            5: [2, 4],
            2: [],
        }
    def forward(self, x):
        B, T, J, D = x.shape
        embedding = getattr(self, f"embedding{J}").view(1, 1, J, D)
        x = x + embedding
        x[:,:, self.map_right[J], :] += self.direc
        x[:,:, self.map_left[J], :] -= self.direc
        return self.dropout(x)


class LearnablePositionalEncoding2D(nn.Module):
    def __init__(self, shape: list|tuple, d_model, dropout=0.1):
        super(LearnablePositionalEncoding2D, self).__init__()
        self.position_embeddings = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(shape[0], shape[1], d_model)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, embeddings):
        # embeddings: [B, L, J, D]
        B, L, J, D = embeddings.shape
        position_embeddings = self.position_embeddings[:L, :J]
        position_embeddings = self.dropout(position_embeddings)
        embeddings = embeddings + position_embeddings.view(1, L, J, D)
        return embeddings

class RotaryPositionEmbedding(nn.Module):
    def __init__(self, head_dim, max_seq_len=64):
        super().__init__()
        self.dim = head_dim
        self.max_seq_len = max_seq_len
        sinusoidal_pos = self._init_sinusoidal_pos()
        self.register_buffer('sinusoidal_pos', sinusoidal_pos)
    
    def _init_sinusoidal_pos(self):
        # build sinusoidal position encoding
        position = torch.arange(self.max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, self.dim, 2).float() * (-math.log(10000.0) / self.dim))
        sinusoidal_pos = torch.zeros((1, self.max_seq_len, self.dim))
        sinusoidal_pos[0, :, 0::2] = torch.sin(position * div_term)
        sinusoidal_pos[0, :, 1::2] = torch.cos(position * div_term)
        return sinusoidal_pos
    
    def forward(self, q, k):
        # q, k: [batch_size, seq_len, hidden_size]
        batch_size, seq_len, hidden_size = q.shape
        
        # reshape to [batch_size, seq_len, num_heads, head_dim]
        head_dim = self.dim
        num_heads = hidden_size // head_dim
        
        q = q.reshape(batch_size, seq_len, num_heads, head_dim)
        k = k.reshape(batch_size, seq_len, num_heads, head_dim)
        
        # get sinusoidal position encoding for current sequence length
        sinusoidal_pos = self.sinusoidal_pos[:, :seq_len, :]
        
        # extract and repeat sine and cosine parts
        cos_pos = sinusoidal_pos[..., None, 1::2].repeat_interleave(2, dim=-1) # [1, seq_len, 1, head_dim]
        sin_pos = sinusoidal_pos[..., None, ::2].repeat_interleave(2, dim=-1) # [1, seq_len, 1, head_dim]
        
        # apply RoPE transformation
        q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=4).view(q.shape)
        q = q * cos_pos + q2 * sin_pos
        
        k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=4).view(k.shape)
        k = k * cos_pos + k2 * sin_pos

        q = q.view(batch_size, seq_len, num_heads * head_dim)
        k = k.view(batch_size, seq_len, num_heads * head_dim)
        
        return q, k
