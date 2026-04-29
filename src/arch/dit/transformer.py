import torch, math
import torch.nn as nn
import numpy as np
from typing import Optional
from torch import Tensor
from einops import rearrange, repeat
from dataclasses import dataclass
from .embedding import RotaryPositionEmbedding


def get_act_fn(act_fn):
    if act_fn == "relu":
        return nn.ReLU()
    elif act_fn == "gelu":
        return nn.GELU()
    elif act_fn == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unknown activation function: {act_fn}")


# modified from: https://github.com/black-forest-labs/flux/blob/main/src/flux/modules/layers.py#L136
@dataclass
class ModulationOut:
    shift: Tensor
    scale: Tensor
    gate: Tensor
    
    def reshape(self, pattern: str, *args, **kwargs):
        return ModulationOut(
            shift = rearrange(self.shift, pattern, *args, **kwargs),
            scale = rearrange(self.scale, pattern, *args, **kwargs),
            gate = rearrange(self.gate, pattern, *args, **kwargs),
        )

    def repeat(self, pattern: str, *args, **kwargs):
        return ModulationOut(
            shift = repeat(self.shift, pattern, *args, **kwargs),
            scale = repeat(self.scale, pattern, *args, **kwargs),
            gate = repeat(self.gate, pattern, *args, **kwargs),
        )
        

class Modulation(nn.Module):
    def __init__(self, dim: int, num: int = 1):
        super().__init__()
        self.multiplier = num * 3
        self.lin = nn.Linear(dim, self.multiplier * dim, bias=True)

    def forward(self, vec: Tensor, n_dim: int = 1) -> tuple[ModulationOut, ModulationOut | None]:
        # vec: [B, D]
        assert n_dim >= 1, f'n_dim should be >= 1, got {n_dim}'
        out = self.lin(nn.functional.silu(vec))
        out = out[(...,) + (None,) * n_dim].transpose(1, -1)
        out = out.chunk(self.multiplier, dim=-1)

        return (ModulationOut(*out[i*3:(i+1)*3]) for i in range(self.multiplier // 3))

@torch.profiler.record_function("attention")
def attention(q: Tensor, k: Tensor, v: Tensor, q_pe: Tensor | None = None, k_pe: Tensor | None = None, attn_mask: Tensor | None = None, dropout: float = 0.0, heads: int = 1, is_split: bool = False) -> Tensor:
    # q, k: [batch_size, seq_len, head_dim]
    # q_pe, k_pe: [batch_size, seq_len, head_dim]
    # attn_mask: [batch_size, seq_len, seq_len]
    if not is_split:
        q = rearrange(q, 'B L (H D) -> B H L D', H=heads)
        k = rearrange(k, 'B K (H D) -> B H K D', H=heads)
        v = rearrange(v, 'B K (H D) -> B H K D', H=heads)
        attn_mask = repeat(attn_mask, 'B N M -> B H N M', H=heads)
    q_pe = q_pe.unsqueeze(1)
    k_pe = k_pe.unsqueeze(1)
    if q_pe is not None and k_pe is not None:
        q, k = apply_rope(q, k, q_pe, k_pe)
    x = torch.nn.functional.scaled_dot_product_attention(
        q, k, v,
        attn_mask=~attn_mask,
        dropout_p = dropout,
    )
    x = rearrange(x, "B H L D -> B L (H D)")
    return x

def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=pos.dtype, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "... d (i j) -> ... d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_q: Tensor, freqs_k: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_q[..., 0] * xq_[..., 0] + freqs_q[..., 1] * xq_[..., 1]
    xk_out = freqs_k[..., 0] * xk_[..., 0] + freqs_k[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1, with_outproj=True):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.dropout = nn.Dropout(dropout)
        self.q_norm = RMSNorm(d_model)
        self.k_norm = RMSNorm(d_model)
        self.Wq = nn.Linear(d_model, d_model)
        self.Wkv = nn.Linear(d_model, 2 * d_model)
        self.with_outproj = with_outproj
        if with_outproj:
            self.Wo = nn.Linear(d_model, d_model)

    def forward(self, target, memory, src_mask=None, key_padding_mask=None, q_rope=None, k_rope=None):
        # q, k, v: [batch_size, seq_len, hidden_size]
        # src_mask: [batch_size, seq_len, seq_len]
        # src_key_padding_mask: [batch_size, seq_len]
        B, N, _ = target.shape
        B, M, _ = memory.shape

        q = self.Wq(target)
        k, v = self.Wkv(memory).chunk(2, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k)

        mask = torch.zeros(B, N, M).to(target.device)
        if src_mask is not None:
            assert src_mask.dim() == 3, f'src_mask should be 3D, got {src_mask.dim()}D'
            mask = mask.masked_fill(src_mask, 1)
        if key_padding_mask is not None:
            assert key_padding_mask.dim() == 2, f'key_padding_mask should be 2D, got {key_padding_mask.dim()}D'
            mask = mask.masked_fill(rearrange(key_padding_mask, 'b k -> b 1 k'), 1)
        out = attention(
            q, k, v,
            q_rope,
            k_rope,
            mask.bool(),
            self.dropout.p * self.training,
            heads = self.nhead,
        )
        if self.with_outproj:
            out = self.Wo(out)
        return out


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_intermediate):
        super().__init__()
        self.gate = nn.Linear(d_model, d_intermediate)
        self.linear1 = nn.Linear(d_model, d_intermediate)
        self.linear2 = nn.Linear(d_intermediate, d_model)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.act(self.gate(x) * self.linear1(x))
        x = self.linear2(x)
        return x

class DoubleStreamBlock(nn.Module):
    """ Pure
    """
    def __init__(
        self, latent_dim, ff_dim, 
        n_heads, dropout=0.3, 
    ):
        super().__init__()
        self.dropout = dropout
        self.n_heads = n_heads

        motion_dim = latent_dim
        self.motion_mod = Modulation(latent_dim, num=2) # attn + ffn
        # motion attention (spatial + temporal)
        self.motion_norm = nn.LayerNorm(motion_dim, elementwise_affine=False, eps=1e-6)
        self.motion_q_norm = RMSNorm(motion_dim)
        self.motion_k_norm = RMSNorm(motion_dim)
        self.motion_qkv = nn.Linear(motion_dim, 3 * motion_dim)
        self.motion_out = nn.Linear(motion_dim, motion_dim)


        # text attention
        self.text_mod = Modulation(latent_dim, num=2) # attn + ffn
        self.text_norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.text_q_norm = RMSNorm(latent_dim)
        self.text_k_norm = RMSNorm(latent_dim)
        self.text_qkv = nn.Linear(latent_dim, 3 * latent_dim)
        self.text_out = nn.Linear(latent_dim, latent_dim)

        # ffn
        self.motion_ffn_norm = nn.LayerNorm(motion_dim, elementwise_affine=False, eps=1e-6)
        self.motion_ffn = nn.Sequential(
            nn.Linear(motion_dim, ff_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ff_dim, motion_dim),
            nn.Dropout(dropout),
        )
        self.text_ffn_norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.text_ffn = nn.Sequential(
            nn.Linear(latent_dim, ff_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ff_dim, latent_dim),
            nn.Dropout(dropout),
        )
    
    @torch.profiler.record_function("double_stream_block_v4")
    def forward(self, motion, text, cond, mask=None, pe=None):
        # motion: [B, T, D]
        # text: [B, N, D]
        # cond: [B, D]
        # motion_mask: [B, T]
        # text_mask: [B, N]
        # motion_rope: [B, T, J, d_head, 2, 2]
        # text_rope: [B, N, d_head, 2, 2]

        N = text.size(1)
        B, T, D = motion.size()
        # motion = rearrange(motion, 'b t j d -> b (t j) d')

        # modulation
        m_mod_attn, m_mod_ff = self.motion_mod(cond, n_dim=1)
        t_mod_attn, t_mod_ff = self.text_mod(cond, n_dim=1)

        # temporal attention
        motion_modulated = self.motion_norm(motion)
        motion_in = (1 + m_mod_attn.scale) * motion_modulated + m_mod_attn.shift
        motion_q, motion_k, motion_v = self.motion_qkv(motion_in).chunk(3, dim=-1)
        motion_q = self.motion_q_norm(motion_q)
        motion_k = self.motion_k_norm(motion_k)

        text_modulated = self.text_norm(text)
        text_in = (1 + t_mod_attn.scale) * text_modulated + t_mod_attn.shift
        text_q, text_k, text_v = self.text_qkv(text_in).chunk(3, dim=-1)
        text_q = self.text_q_norm(text_q)
        text_k = self.text_k_norm(text_k)

        _q, _k, _v = torch.cat([motion_q, text_q], dim=1), torch.cat([motion_k, text_k], dim=1), torch.cat([motion_v, text_v], dim=1)
        motion_out, text_out = attention(
            _q, _k, _v,
            q_pe = pe,
            k_pe = pe,
            # attn_mask = repeat(torch.cat([motion_mask, text_mask], dim=1), 'b n -> b m n', m = T + N),
            attn_mask = repeat(mask, 'b n -> b m n', m = T + N),
            dropout = self.dropout * self.training,
            heads = self.n_heads,
        ).split([T, N], dim=1)
        motion_out = self.motion_out(motion_out)
        text_out = self.text_out(text_out)
        motion = motion + m_mod_attn.gate * motion_out
        text = text + t_mod_attn.gate * text_out

        # feed-forward
        motion_ff_in = self.motion_ffn_norm(motion)
        motion_ff_in = (1 + m_mod_ff.scale) * motion_ff_in + m_mod_ff.shift
        motion_ff_out = self.motion_ffn(motion_ff_in)
        motion = motion + m_mod_ff.gate * motion_ff_out

        text_ff_in = self.text_ffn_norm(text)
        text_ff_in = (1 + t_mod_ff.scale) * text_ff_in + t_mod_ff.shift
        text_ff_out = self.text_ffn(text_ff_in)
        text = text + t_mod_ff.gate * text_ff_out

        # if J is not None:
        #     motion = rearrange(motion, 'b (t j) d -> b t j d', t=T, j=J)

        return motion, text

class SingleStreamBlock(nn.Module):
    def __init__(self, latent_dim, ff_dim, n_heads, dropout=0.3):
        super().__init__()

        motion_dim = latent_dim
        ff_dim = ff_dim
        self.motion_norm = nn.LayerNorm(motion_dim, elementwise_affine=False, eps=1e-6)
        self.motion_mod = Modulation(latent_dim, num=1)
        self.self_attn = MultiHeadAttention(motion_dim, n_heads, dropout, with_outproj=False)
    
        self.linear1 = nn.Linear(motion_dim, ff_dim)
        self.act = nn.GELU(approximate="tanh")
        self.linear2 = nn.Linear(ff_dim + motion_dim, motion_dim)
        self.dropout = nn.Dropout(dropout)

    @torch.profiler.record_function("single_stream_block_v6")
    def forward(self, motion, cond, mask=None, pe=None):
        B, T, D = motion.size()
        mod, = self.motion_mod(cond, n_dim=1)

        motion_mod = self.motion_norm(motion)
        motion_mod = (1 + mod.scale) * motion_mod + mod.shift

        attn_out = self.self_attn(
            motion_mod, motion_mod, 
            key_padding_mask = mask,
            q_rope = pe,
            k_rope = pe
        )
        # attn_out = rearrange(attn_out, 'b (t j) d -> b t j d', j=J)

        output = self.linear2(torch.cat([attn_out, self.act(self.linear1(motion_mod))], dim=-1))
        output = self.dropout(output)

        return motion + mod.gate * output
    
class LastLayer(nn.Module):
    def __init__(self, hidden_size: int, output_size: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, output_size, bias=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, motion: Tensor, vec: Tensor) -> Tensor:
        # motion: [B, T, J, D]
        # vec: [B, D]
        B, T, D = motion.size()
        shift, scale = self.adaLN_modulation(vec).chunk(2, dim=1)
        motion = (1 + scale[:, None, :]) * self.norm_final(motion) + shift[:, None, :]
        motion = self.linear(motion)
        return motion

class TimeBlock(nn.Module):
    def __init__(self, latent_dim: int, time_patch_size: int, output_size: int):
        super().__init__()
        self.time_patch_size = time_patch_size
        self.unpatching = nn.Linear(latent_dim, latent_dim * time_patch_size)
        self.mod = Modulation(latent_dim, num=2)
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False, eps=1e-6)
        self.W_qkv = nn.Linear(latent_dim, latent_dim * 3)
        self.q_norm = RMSNorm(latent_dim)
        self.k_norm = RMSNorm(latent_dim)
        self.linear1 = nn.Linear(latent_dim, latent_dim)
        self.act = nn.GELU(approximate="tanh")
        self.linear2 = nn.Linear(latent_dim * 2, output_size)
    
    
    def forward(self, x: Tensor, cond=None, pe=None, mask=None) -> Tensor:
        # x: [B, T, J, (tp, d)]
        # pe: [B, tp, J, d]
        # mask: [B, T*tp]
        B, T, J, D = x.size()
        mod, = self.mod(cond, n_dim=1)
        mod = mod.repeat('b 1 d -> (b t) 1 d', t=T)

        x = self.unpatching(x)
        x = rearrange(x, 'b t j (tp d) -> (b t) (tp j) d', tp=self.time_patch_size)
        x = self.norm(x)
        x = (1 + mod.scale) * x + mod.shift

        q, k, v = self.W_qkv(x).chunk(3, dim=-1)
        q = self.q_norm(q)
        k = self.k_norm(k)
        if pe is not None:
            pe = repeat(pe, 'b tp j ... -> (b t) (tp j) ...', t=T)
        if mask is not None:
            mask = repeat(mask, 'b (t tp) -> (b t) (tp j)', tp=self.time_patch_size, j=J)
        attn_out = attention(
            q, k, v,
            q_pe = pe,
            k_pe = pe,
            attn_mask = mask,
        )
        output = self.linear2(torch.cat([attn_out, self.act(self.linear1(x))], dim=-1))
        output = rearrange(output, '(b t) (tp j) d -> b (t tp) j d', tp=self.time_patch_size, j=J)
        return x + output

class MotionFlux(nn.Module):
    def __init__(self, latent_dim, n_heads, n_double_layers, n_single_layers, output_size, dropout=0.3):
        super().__init__()
        
        # transformer encoder
        self.double_blocks = nn.ModuleList()
        self.single_blocks = nn.ModuleList()
        self.last_layer = LastLayer(latent_dim, output_size)
        
        max_len = 50
        pe = torch.zeros(max_len, latent_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, latent_dim, 2).float() * (-np.log(10000.0) / latent_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('q_pe', pe) #[max_len, latent_dim]

        for i in range(n_double_layers):
            self.double_blocks.append(DoubleStreamBlock(latent_dim=latent_dim, ff_dim=latent_dim*4, n_heads=n_heads, dropout=dropout))
        for i in range(n_single_layers):
            self.single_blocks.append(SingleStreamBlock(latent_dim=latent_dim, ff_dim=latent_dim*4, n_heads=n_heads, dropout=dropout))

    # def forward(self, x, vec, word_emb, sa_mask=None, ca_mask=None, motion_rope=None, text_rope=None, motion_rope_st=None):
    def forward(self, x, vec, word_emb, sa_mask, ca_mask, pe=None):
        """
        x: [B, T, J, D]
        vec: [B, D]
        word_emb: [B, N, D]
        sa_mask: [B, T]
        ca_mask: [B, N]
        """
        # B, T, J, D = x.size()
        mask = torch.cat([sa_mask, ca_mask], dim=1)
        for block in self.double_blocks:
            x, word_emb = block(x, word_emb, vec, mask=mask, pe=pe)

        xt = torch.cat([x, word_emb], dim=1)
        for block in self.single_blocks:
            xt = block(xt, vec, mask=mask, pe=pe)

        x = xt[:, :x.size(1), ...]
        x = self.last_layer(x, vec)

        return x