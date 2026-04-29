import torch

def adj_vec2mat(adj_vec):
    out_size = len(adj_vec)
    in_size = max([max(adj) for adj in adj_vec]) + 1
    adj_mat = torch.zeros(in_size, out_size)
    for j, adjs in enumerate(adj_vec):
        for i in adjs:
            adj_mat[i, j] = 1
    return adj_mat

def edges_to_adj_mat(edges, self_loop=True, centers=[]):
    out_size = max([max(edge) for edge in edges]) + 1
    in_size = out_size
    adj_mat = torch.zeros(in_size, out_size)
    for u, v in edges:
        adj_mat[u, v] = 1
        adj_mat[v, u] = 1
    if self_loop:
        for i in range(in_size):
            adj_mat[i, i] = 1
    for center in centers:
        adj_mat[center, :] = 1
        adj_mat[:, center] = 1
    return adj_mat

# id | vertical (up, down) | horizontal (left, middle, right) | parent (graph feature)
# 0: root           | 0 | 0 | 0
# 3: torso          | 1 | 0 | 1
# 4: left upper     | 1 | 1 | 2
# 5: right upper    | 1 | -1 | 2
# 1: left lower     | -1 | 1 | 1
# 2: right lower    | -1 | -1 | 1
joint_pos_id = {
    6: torch.tensor([
        [0, 0, 0], # 0 root
        [1, 0, 1], # 3 torso
        [1, 1, 2], # 4 left upper
        [1, -1, 2], # 5 right upper
        [-1, 1, 1], # 1 left lower
        [-1, -1, 1], # 2 right lower
    ]),
}

out_ward = {
    22: [
        (0, 1), (0, 2), (0, 3),
        (1, 4),
        (2, 5),
        (3, 6),
        (4, 7),
        (5, 8),
        (6, 9),
        (7, 10),
        (8, 11),
        (9, 12), (9, 13), (9, 14),
        (12, 15),
        (13, 16),
        (14, 17),
        (16, 18),
        (17, 19),
        (18, 20),
        (19, 21),
    ],
    21: [
        (0, 1), (0, 11), (0, 16),
        (1, 2),
        (2, 3),
        (3, 4), (3, 5), (3, 8),
        (5, 6),
        (6, 7),
        (8, 9),
        (9, 10),
        (11, 12),
        (12, 13),
        (13, 14),
        (14, 15),
        (16, 17),
        (17, 18),
        (18, 19),
        (19, 20),
    ],
    14: [
        (0, 1), (0, 8), (0, 11),
        (1, 2),
        (2, 3), (2, 4), (2, 6),
        (4, 5),
        (6, 7),
        (8, 9), (9, 10),
        (11, 12), (12, 13),
    ],
    6: [
        (0, 1), (0, 4), (0, 5),
        (1, 2),
        (1, 3),
    ],
}


"""
14:
    3
5 4 2 6 7
    1
    0
  8   11
  9   12
 10   13


6:

2-1-3
  0
 / \
4   5


"""

pool_adj = {
    '22->14': adj_vec2mat([
        [0, 1, 2, 3],
        [0, 3 ,6, 9],
        [6, 9, 12, 13, 14],
        [9, 12, 15],
        [9, 14, 17],
        [17, 19, 21],
        [9, 13, 16],
        [16, 18, 20],
        [0, 2, 5],
        [2, 5, 8], 
        [5, 8, 11],
        [0, 1 ,4],
        [1, 4, 7],
        [4, 7, 10],
    ]),
    '21->14': adj_vec2mat([
        [0, 1, 11, 16],
        [0, 1, 2, 3],
        [2, 3, 4, 5, 8],
        [3, 4],
        [3, 5, 6],
        [5, 6, 7],
        [3, 8, 9],
        [8, 9, 10],
        [0, 11, 12],
        [12, 13, 14],
        [14, 15],
        [0, 16, 17],
        [17, 18, 19],
        [19, 20],
    ]),
    '14->6': adj_vec2mat([
        [0, 1, 8, 11],
        [1, 2, 3, 4, 6],
        [2, 4, 5],
        [2, 6, 7],
        [0, 8, 9, 10],
        [0, 11, 12, 13],
    ]),
    '14->22': adj_vec2mat([
        [0, 1, 2, 3],
        [0, 3 ,6, 9],
        [6, 9, 12, 13, 14],
        [9, 12, 15],
        [9, 14, 17],
        [17, 19, 21],
        [9, 13, 16],
        [16, 18, 20],
        [0, 2, 5],
        [2, 5, 8], 
        [5, 8, 11],
        [0, 1 ,4],
        [1, 4, 7],
        [4, 7, 10],
    ]).T,
    '14->21': adj_vec2mat([
        [0, 1, 11, 16],
        [0, 1, 2, 3],
        [2, 3, 4, 5, 8],
        [3, 4],
        [3, 5, 6],
        [5, 6, 7],
        [3, 8, 9],
        [8, 9, 10],
        [0, 11, 12],
        [12, 13, 14],
        [14, 15],
        [0, 16, 17],
        [17, 18, 19],
        [19, 20],
    ]).T,
    '6->14': adj_vec2mat([
        [0, 1, 8, 11],
        [1, 2, 3, 4, 6],
        [2, 4, 5],
        [2, 6, 7],
        [0, 8, 9, 10],
        [0, 11, 12, 13],
    ]).T,
}


# https://github.com/stnoah1/infogcn/blob/master/graph/ucla.py
import numpy as np
import src.utils.graph_tools as tools

# num_node = 20
# self_link = [(i, i) for i in range(num_node)]
# inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
# outward = [(j, i) for (i, j) in inward]
# neighbor = inward + outward


class Graph:
    def __init__(self, joints_num, labeling_mode='spatial', scale=1):
        self.num_node = joints_num
        self.self_link = [(i, i) for i in range(joints_num)]
        self.inward = [(j, i) for (i, j) in out_ward[joints_num]]
        self.outward = out_ward[joints_num]
        self.neighbor = self.inward + self.outward
        self.A = self.get_adjacency_matrix(labeling_mode)
        self.A_outward_binary = tools.get_adjacency_matrix(self.outward, self.num_node)
        self.A_binary = tools.edge2mat(self.neighbor, self.num_node)
        self.A_norm = tools.normalize_adjacency_matrix(self.A_binary + 2*np.eye(self.num_node))
        self.A_binary_K = tools.get_k_scale_graph(scale, self.A_binary)

    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(self.num_node, self.self_link, self.inward, self.outward)
        else:
            raise ValueError()
        return A