import torch
import torch.nn as nn
from einops import rearrange


def interpolate(x, scale_factor, type='half-diff', mode='linear'):
    """ 
    interpolate the motion. 
    Note that we use the cumulative sum of the motion to interpolate, and then take the difference to get the interpolated motion.

    Args
    ----
        - x: [B, T, J, D] or [B, T, D]

    Returns
    -------
        - x: [B, T, J, D] or [B, T, D]
    """
    assert type in ['cum-diff', 'straight', 'half-diff'], f'{type} is not supported'
    assert x.ndim in [3, 4], f'x should have 3 or 4 dimensions, but got {x.ndim}'
    if type == 'half-diff':
        s_d = x.shape[-1] // 4 * 3
        assert x.shape[-1] % 4 == 0, f'when using half-diff, x should have a length that is a multiple of 4, but got {x.shape[-1]}'
        return torch.cat([ # for v, and other types of data
            interpolate(x[..., :s_d], scale_factor, type='straight', mode=mode),
            interpolate(x[..., s_d:], scale_factor, type='cum-diff', mode=mode),
        ], dim=-1)
    remove_j = False
    if x.ndim == 3:
        x = x.unsqueeze(-2)
        remove_j = True
    B, T, J, D = x.size()
    x = rearrange(x, 'b t j d -> (b j) d t')
    if type == 'cum-diff':
        x = torch.cumsum(x, dim=-1)
    x = torch.nn.functional.interpolate(x, scale_factor=scale_factor, mode=mode)
    if type == 'cum-diff':
        x = torch.diff(x, dim=-1, prepend=torch.zeros_like(x[..., :1]))
    x = rearrange(x, '(b j) d t -> b t j d', b=B)
    if remove_j:
        x = x.squeeze(-2)
    return x

