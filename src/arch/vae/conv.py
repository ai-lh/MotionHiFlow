import torch, math
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Rearrange

def get_activation(name):
    if name.lower() == "relu":
        return nn.ReLU()
    elif name.lower() == "gelu":
        return nn.GELU()
    elif name.lower() == "silu":
        return nn.SiLU()
    else:
        raise ValueError(f"Unknown activation function: {name}")


def get_norm(name, dim):
    if name.lower() == "layer" or name.lower() == "ln":
        return nn.Sequential(
            Rearrange('b d t j -> b t j d'),
            nn.LayerNorm(dim),
            Rearrange('b t j d -> b d t j'),
        )
    elif name.lower() == "batch" or name.lower() == "bn":
        return nn.BatchNorm2d(dim)
    elif name.lower() == "group" or name.lower() == "gn":
        return nn.GroupNorm(32, dim)
    elif name.lower() == "none":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown normalization function: {name}")

# https://github.com/lshiwjx/2s-AGCN/blob/master/model/aagcn.py

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_in')
    nn.init.constant_(conv.bias, 0)

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, activation="relu", causal=True):
        super().__init__()
        pad = int((kernel_size - 1) / 2)
        if causal:
            pad = 0
        self.causal = causal
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), dilation=(dilation, 1))

        conv_init(self.conv)

    @torch.profiler.record_function("unit_tcn")
    def forward(self, x):
        if self.causal:
            x = torch.nn.functional.pad(x, (0, 0, (self.kernel_size - 1) * self.dilation, 0))
        return self.conv(x)

class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, coff_embedding=4, num_subset=3, norm="none", activation="relu"):
        super().__init__()
        inter_channels = out_channels // coff_embedding
        self.inter_c = inter_channels
        self.PA = nn.Parameter(torch.from_numpy(A).to(torch.float32))
        nn.init.constant_(self.PA, 1e-6)
        self.A = nn.Parameter(torch.from_numpy(A).to(torch.float32), requires_grad=False)
        self.num_subset = num_subset

        self.conv_a = nn.ModuleList()
        self.conv_b = nn.ModuleList()
        self.conv_d = nn.ModuleList()
        for i in range(self.num_subset):
            self.conv_a.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_b.append(nn.Conv2d(in_channels, inter_channels, 1))
            self.conv_d.append(nn.Conv2d(in_channels, out_channels, 1))

        if in_channels != out_channels:
            self.down = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.down = lambda x: x

        self.soft = nn.Softmax(-2)
        self.act = get_activation(activation)
        self.norm = get_norm(norm, out_channels)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
        for i in range(self.num_subset):
            conv_branch_init(self.conv_d[i], self.num_subset)

    @torch.profiler.record_function("unit_gcn")
    def forward(self, x):
        N, C, T, V = x.size()
        A = self.A.to(x.device)
        A = A + self.PA

        y = None
        for i in range(self.num_subset):
            A1 = self.conv_a[i](x).permute(0, 3, 1, 2).reshape(N, V, self.inter_c * T)
            A2 = self.conv_b[i](x).reshape(N, self.inter_c * T, V)
            A1 = self.soft(torch.matmul(A1, A2) / A1.size(-1))  # N V V
            A1 = A1 + A[i]
            A2 = x.reshape(N, C * T, V)
            z = self.conv_d[i](torch.matmul(A2, A1).reshape(N, C, T, V))
            y = z + y if y is not None else z

        y += self.down(x)
        y = self.norm(y)
        return self.act(y)


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, causal=True, dilation=1, activation="relu", norm="none"):
        super().__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, activation=activation)
        self.tcn1 = unit_tcn(out_channels, out_channels, stride=stride, causal=causal, dilation=dilation, activation=activation)
        self.act = get_activation(activation)
        self.norm = get_norm(norm, out_channels)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride, activation=activation)

    def forward(self, x):
        x = self.tcn1(self.gcn1(x)) + self.residual(x)
        x = self.norm(x)
        return self.act(x)

class GraphPool(nn.Module):
    def __init__(self, A):
        super().__init__()
        # A: [in, out]
        A = A / A.sum(dim=-1, keepdim=True)
        self.A = torch.nn.Parameter(A, requires_grad=False)

    def forward(self, x):
        return torch.einsum('bdtv, vo -> bdto', x, self.A)
