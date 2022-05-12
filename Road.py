import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

# class Road(nn.Module):
#     def __init__(self):
#         super(Road, self).__init__()
#         self.build()
#
#     def build(self):
#         self.embedding = nn.Embedding(128*128, 32)
#         emb_vectors = np.load('Config/embedding_128.npy')
#         self.embedding.weight.data.copy_(torch.from_numpy(emb_vectors))
#         self.process_coords = nn.Linear(2+32, 32)
#         self.conv = nn.Conv1d(32, 64, 3)
#
# #        for module in self.modules():
# #            if type(module) is not nn.Embedding:
# #                continue
# #            nn.init.uniform_(module.state_dict()['weight'], a=-1, b=1)
#
#     def forward(self, traj):
#         # road network structure layer
#         lngs = torch.unsqueeze(traj['lngs'], dim = 2)
#         lats = torch.unsqueeze(traj['lats'], dim = 2)
#         grid_ids = torch.unsqueeze(traj['grid_id'].long(), dim = 2)
#         grids = torch.squeeze(self.embedding(grid_ids))
#
#         locs = torch.cat([lngs, lats, grids], dim = 2)
#         locs = self.process_coords(locs)
#         locs = F.tanh(locs)
#
#         return locs


class Road(nn.Module):
    def __init__(self):
        super(Road, self).__init__()
        self.build()

    def build(self):
        self.dim = 32
        self.len = 135
        self.num_filter = 32
        self.kernel_size = 3
        self.plus = 8
        self.embedding = nn.Embedding(128 * 128, 32)
        emb_vectors = np.load('data/embedding_128.npy')
        self.embedding.weight.data.copy_(torch.from_numpy(emb_vectors))
        self.process_coords = nn.Linear(2 + 32, 32)
    #        for module in self.modules():
    #            if type(module) is not nn.Embedding:
    #                continue
    #            nn.init.uniform_(module.state_dict()['weight'], a=-1, b=1)

    def forward(self, traj):
        # road network structure layer
        lngs = torch.unsqueeze(traj['lngs'], dim=2)
        lats = torch.unsqueeze(traj['lats'], dim=2)
        grid_ids = torch.unsqueeze(traj['grid_id'].long(), dim=2)
        grids = torch.squeeze(self.embedding(grid_ids))

        # print('lngs:', lngs.shape)
        # print('lats:', lats.shape)
        # print('grid_ids:', grid_ids.shape)
        # print('grids:', grids.shape)

        locs = torch.cat([lngs, lats, grids], dim=2)
        locs = self.process_coords(locs)
        locs = F.tanh(locs)

        return locs
