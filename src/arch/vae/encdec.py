import torch
import math
import torch.nn as nn
from einops import rearrange
from .conv import TCN_GCN_unit, get_activation, GraphPool, unit_tcn, get_norm
from src.utils.skeleton import Graph, pool_adj


class Motion1Dto2D(nn.Module):
    def __init__(self, pose_dim, width, activation='relu', dataset_name='t2m'):
        super().__init__()

        self.pose_dim = pose_dim
        self.joints_num = (self.pose_dim + 1) // 12
        self.width = width  
        if dataset_name == 't2m':
            self.fid = [7, 10, 8, 11] # fid_l [7, 10] and fid_r [8, 11] ; refer to utils/motion_process.py
        elif dataset_name == 'kit':
            self.fid = [19, 20, 14, 15] # fid_l [19, 20] and fid_r [14, 15]
        elif dataset_name == 'snap':
            self.fid = [18, 19, 22, 23] # fid_l [18, 19] and fid_r [22, 23]
        else:
            raise ValueError(f'Dataset {dataset_name} not supported')

        self.weight1 = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(self.joints_num, 14, width)))
        self.bias1 = torch.nn.Parameter(torch.zeros(self.joints_num, width))
        self.act = get_activation(activation)
        self.weight2 = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(self.joints_num, width, width)))
        self.bias2 = torch.nn.Parameter(torch.zeros(self.joints_num, width))

    def reset_parameters(self) -> None:
        fan_in1, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight1.view(self.joints_num * 14, -1))
        fan_in2, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight2.view(self.joints_num * self.width, -1))
        bound1 = 1 / math.sqrt(fan_in1) if fan_in1 > 0 else 0
        bound2 = 1 / math.sqrt(fan_in2) if fan_in2 > 0 else 0
        torch.nn.init.uniform_(self.bias1, -bound1, bound1)
        torch.nn.init.uniform_(self.bias2, -bound2, bound2)

    def forward(self, x):
        """
        x: [bs, nframes, pose_dim]
        
        nfeats = 12 * self.joints_num + 1
            - root_rot_velocity (B, seq_len, 1)
            - root_linear_velocity (B, seq_len, 2)
            - root_y (B, seq_len, 1)
            - ric_data (B, seq_len, (joint_num - 1)*3)
            - rot_data (B, seq_len, (joint_num - 1)*6)
            - local_velocity (B, seq_len, joint_num*3)
            - foot contact (B, seq_len, 4)
        """
        B, T, D = x.size()

        # split
        root, ric, rot, vel, contact = torch.split(x, [4, 3 * (self.joints_num - 1), 6 * (self.joints_num - 1), 3 * self.joints_num, 4], dim=-1)
        ric = ric.reshape(B, T, self.joints_num - 1, 3)
        rot = rot.reshape(B, T, self.joints_num - 1, 6)
        vel = vel.reshape(B, T, self.joints_num, 3)

        joints = [torch.cat([root, vel[:, :, 0]], dim=-1)] # [B, T, 7]]
        # joints = [torch.cat([joints[0], torch.cumsum(vel[:, :, 0], dim=-2)], dim=-1)] # might works when latent down_sample 
        for i in range(1, self.joints_num):
            joints.append(torch.cat([ric[:, :, i - 1], rot[:, :, i - 1], vel[:, :, i]], dim=-1))
        for cidx, jidx in enumerate(self.fid):
            joints[jidx] = torch.cat([joints[jidx], contact[:, :, cidx, None]], dim=-1)
        joints = [
            torch.nn.functional.pad(joint, (0, 14 - joint.shape[-1])) for joint in joints
        ]
        joints = torch.stack(joints, dim=-2) # [B, T, V, D]

        out = torch.einsum('btvd, vdo -> btvo', joints, self.weight1) + self.bias1
        out = self.act(out)
        out = torch.einsum('btvd, vdo -> btvo', out, self.weight2) + self.bias2
        out = rearrange(out, 'b t v d -> b d t v')
        return out # [B, C, T, V]


class Motion2Dto1D(nn.Module):
    def __init__(self, pose_dim, width, activation='relu', dataset_name='t2m'):
        super().__init__()
        
        self.pose_dim = pose_dim
        self.joints_num = (self.pose_dim + 1) // 12
        self.width = width
        if dataset_name == 't2m':
            self.fid = [7, 10, 8, 11] # fid_l [7, 10] and fid_r [8, 11] ; refer to utils/motion_process.py
        elif dataset_name == 'kit':
            self.fid = [19, 20, 14, 15] # fid_l [19, 20] and fid_r [14, 15]
        elif dataset_name == 'snap':
            self.fid = [18, 19, 22, 23] # fid_l [18, 19] and fid_r [22, 23]
        else:
            raise ValueError(f'Dataset {dataset_name} not supported')

        # network components
        self.weight1 = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(self.joints_num, width, width)))
        self.bias1 = torch.nn.Parameter(torch.zeros(self.joints_num, width))
        self.act = get_activation(activation)
        self.weight2 = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.empty(self.joints_num, width, 14)))
        self.bias2 = torch.nn.Parameter(torch.zeros(self.joints_num, 14))

    def reset_parameters(self) -> None:
        fan_in1, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight1.view(self.joints_num * 14, -1))
        fan_in2, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight2.view(self.joints_num * self.width, -1))
        bound1 = 1 / math.sqrt(fan_in1) if fan_in1 > 0 else 0
        bound2 = 1 / math.sqrt(fan_in2) if fan_in2 > 0 else 0
        torch.nn.init.uniform_(self.bias1, -bound1, bound1)
        torch.nn.init.uniform_(self.bias2, -bound2, bound2)

    def forward(self, x):
        """
        x: [bs, nframes, joints_num, latent_dim]
        """
        B, D, T, V = x.size()
        out = torch.einsum('bdtv, vdo -> btvo', x, self.weight1) + self.bias1
        out = self.act(out)
        out = torch.einsum('btvd, vdo -> btvo', out, self.weight2) + self.bias2
        root = out[..., 0, :7]

        ric_list, rot_list, vel_list = [], [], []
        for i in range(1, self.joints_num):
            ric = out[..., i, :3]
            rot = out[..., i, 3:9]
            vel = out[..., i, 9:12]

            ric_list.append(ric)
            rot_list.append(rot)
            vel_list.append(vel)

        contact = [out[..., i, -1] for i in self.fid]

        ric = torch.stack(ric_list, dim=2).reshape(B, T, (V - 1) * 3)
        rot = torch.stack(rot_list, dim=2).reshape(B, T, (V - 1) * 6)
        vel = torch.stack(vel_list, dim=2).reshape(B, T, (V - 1) * 3)
        contact = torch.stack(contact, dim=2).reshape(B, T, len(self.fid))

        motion = torch.cat([
            root[..., :4], # root
            ric, # ric
            rot, # rot
            torch.cat([root[..., 4:], vel], dim=-1), # vel
            contact, # contact
        ], dim=-1)

        return motion

class Encoder(nn.Module):
    def __init__(
        self, num_joints, width=64,
        depth=3, dilation_growth_rate=1,
        activation='relu', ch_mult=None, 
        causal=True, norm="none", 
    ):
        super().__init__()

        num_channels = [width] * len(num_joints)
        if ch_mult is not None:
            assert len(ch_mult) == len(num_joints), f'ch_mult must be the same length as num_joints, but got ch_mult={ch_mult} and num_joints={num_joints}'
            num_channels = [width * m for m in ch_mult]

        # network
        self.layers = nn.ModuleList()
        if num_channels[0] != width:
            self.layers.append(nn.Sequential(
                nn.Conv2d(width, num_channels[0], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            ))

        A = Graph(num_joints[0]).A
        for i in range(len(num_joints) - 1):
            layers = []
            for _ in range(depth - 1):
                layers.append(TCN_GCN_unit(
                    num_channels[i],
                    num_channels[i],
                    A, stride=1,
                    causal=causal,
                    residual=True,
                    activation=activation,
                    dilation=dilation_growth_rate ** i,
                    norm=norm,
                ))
            layers.append(torch.nn.Sequential(
                GraphPool(pool_adj[f'{num_joints[i]}->{num_joints[i + 1]}']),
                nn.Conv2d(num_channels[i], num_channels[i + 1], kernel_size=(3, 1), stride=(2, 1), padding=(1, 0)),
            ))

            A = Graph(num_joints[i + 1]).A
            layers.append(TCN_GCN_unit(
                num_channels[i + 1],
                num_channels[i + 1],
                A, stride=1,
                causal=causal,
                residual=True,
                activation=activation,
                dilation=dilation_growth_rate ** i,
                norm=norm,
            ))
            self.layers.append(nn.Sequential(*layers))
        if num_channels[-1] != width:
            self.layers.append(nn.Sequential(
                nn.Conv2d(num_channels[-1], width, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            ))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x


class Decoder(nn.Module):
    def __init__(
        self, num_joints, width=64, 
        depth=3, dilation_growth_rate=1,
        activation='relu', ch_mult=None, 
        causal=True, norm="none", 
    ):
        super().__init__()

        num_joints = num_joints[::-1]
        num_channels = [width] * len(num_joints)
        if ch_mult is not None:
            assert len(ch_mult) == len(num_joints), f'ch_mult must be the same length as num_joints, but got ch_mult={ch_mult} and num_joints={num_joints}'
            num_channels = [width * m for m in ch_mult]
            num_channels = num_channels[::-1]

        self.layers = nn.ModuleList()
        if num_channels[0] != width:
            self.layers.append(nn.Sequential(
                nn.Conv2d(width, num_channels[0], kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            ))

        A = Graph(num_joints[0]).A
        for i in range(len(num_joints) - 1):
            layers = []
            layers.append(TCN_GCN_unit(
                num_channels[i],
                num_channels[i],
                A, stride=1,
                causal=causal,
                residual=True,
                activation=activation,
                dilation=dilation_growth_rate ** (len(num_joints) - i - 2),
                norm=norm,
            ))
            layers.append(nn.Sequential(
                nn.Upsample(scale_factor=(2, 1), mode='nearest'), # follow previous work
                # nn.Upsample(scale_factor=(2, 1), mode='bilinear'), # better acc (accelerate error)
                unit_tcn(num_channels[i], num_channels[i + 1], kernel_size=3, stride=1, activation=activation, causal=causal),
                GraphPool(pool_adj[f'{num_joints[i]}->{num_joints[i + 1]}']),
            ))
            A = Graph(num_joints[i + 1]).A
            for _ in range(depth - 1):
                layers.append(TCN_GCN_unit(
                    num_channels[i + 1],
                    num_channels[i + 1],
                    A, stride=1,
                    causal=causal,
                    residual=True,
                    activation=activation,
                    dilation=dilation_growth_rate ** (len(num_joints) - i - 2),
                    norm=norm,
                ))

            self.layers.append(nn.Sequential(*layers))
        if num_channels[-1] != width:
            self.layers.append(nn.Sequential(
                nn.Conv2d(num_channels[-1], width, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            ))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x
