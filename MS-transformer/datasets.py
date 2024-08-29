# coding=utf-8
# Copyright 2021, Duong Nguyen
#
# Licensed under the CECILL-C License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.cecill.info
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Customized Pytorch Dataset.
"""

import numpy as np
import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader


class AISDataset(Dataset):
    """Customized Pytorch dataset.
    """

    def __init__(self,
                 l_data,
                 max_seqlen=96,
                 dtype=torch.float32,
                 device=torch.device("cpu")):
        """
        Args
            l_data: list of dictionaries, each element is an AIS trajectory. 
                l_data[idx]["mmsi"]: vessel's MMSI.
                l_data[idx]["traj"]: a matrix whose columns are 
                    [LAT, LON, SOG, COG, TIMESTAMP]
                lat, lon, sog, and cod have been standardized, i.e. range = [0,1).
            max_seqlen: (optional) max sequence length. Default is
        """

        self.max_seqlen = max_seqlen
        self.device = device

        self.l_data = l_data

    def __len__(self):
        return len(self.l_data)

    def __getitem__(self, idx):
        """Gets items.
        
        Returns:
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a
            padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        V = self.l_data[idx]
        m_v = V["traj"][:,:4]# lat, lon, sog, cog前4列
        #         m_v[m_v==1] = 0.9999
        m_v[m_v > 0.9999] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)  # 返回这两个长度中的较小值，序列的实际长度

        target = np.full((1 , 2), np.nan)#创建目的地序列，将最轨迹终点填入
        target[:, 0] = m_v[seqlen - 1][0]
        target[:, 1] = m_v[seqlen - 1][1]
        # m_v = np.hstack([m_v, target])

        seq = np.zeros((self.max_seqlen, 4))
        seq[:seqlen, :] = m_v[:seqlen, :]
        seq = torch.tensor(seq, dtype=torch.float32)  # 转化为torch.tensor

        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.  # 表示这些位置是非padding

        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi = torch.tensor(V["mmsi"], dtype=torch.int)
        time_start = torch.tensor(V["traj"][0, 4], dtype=torch.int)

        return seq, target, mask, seqlen, mmsi, time_start


class AISDataset_new(Dataset):
    """Customized Pytorch dataset.
    """

    def __init__(self,
                 l_data,
                 max_seqlen=96,
                 dtype=torch.float32,
                 device=torch.device("cpu")):
        """
        Args
            l_data: list of dictionaries, each element is an AIS trajectory.
                l_data[idx]["mmsi"]: vessel's MMSI.
                l_data[idx]["traj"]: a matrix whose columns are
                    [LAT, LON, SOG, COG, TIMESTAMP]
                lat, lon, sog, and cod have been standardized, i.e. range = [0,1).
            max_seqlen: (optional) max sequence length. Default is
        """

        self.max_seqlen = max_seqlen
        self.device = device

        self.l_data = l_data

    def __len__(self):
        return len(self.l_data)

    def __getitem__(self, idx):
        """Gets items.

        Returns:
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a
            padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        V = self.l_data[idx]
        m_v = V["traj"][:, :4]  # lat, lon, sog, cog前4列
        weight_v = V["traj"][:, 6:]
        #         m_v[m_v==1] = 0.9999
        m_v[m_v > 0.9999] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)  # 返回这两个长度中的较小值，序列的实际长度

        target = np.full((1, 2), np.nan)  # 创建目的地序列，将最轨迹终点填入
        target[:, 0] = m_v[seqlen - 1][0]
        target[:, 1] = m_v[seqlen - 1][1]

        seq = np.zeros((self.max_seqlen, 4))
        seq[:seqlen, :] = m_v[:seqlen, :] # (141,4)
        seq = torch.tensor(seq, dtype=torch.float32)  # 转化为torch.tensor

        seq_weight = np.zeros((self.max_seqlen, 5))
        seq_weight[:seqlen, :] = weight_v[:seqlen, :]  # (141,5)

        # 填充seq_weight后续为0的位置
        last_non_zero_weight = seq_weight[seqlen - 1]
        if seqlen < self.max_seqlen:
            for i in range(seqlen, self.max_seqlen):
                seq_weight[i] = last_non_zero_weight
        seq_weight = torch.tensor(seq_weight, dtype=torch.float32)  # 转化为torch.tensor

        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.  # 表示这些位置是非padding

        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi = torch.tensor(V["mmsi"], dtype=torch.int)
        # time_start = torch.tensor(V["traj"][0, 4], dtype=torch.int)
        # time_start = V["traj"][0, 4].clone().detach().int()
        return seq, target, mask, seqlen, mmsi, seq_weight


class AISDataset_sim(Dataset):
    """Customized Pytorch dataset.
    """

    def __init__(self,
                 l_data,
                 max_seqlen=96,
                 dtype=torch.float32,
                 device=torch.device("cpu")):
        """
        Args
            l_data: list of dictionaries, each element is an AIS trajectory.
                l_data[idx]["mmsi"]: vessel's MMSI.
                l_data[idx]["traj"]: a matrix whose columns are
                    [LAT, LON, SOG, COG, TIMESTAMP]
                lat, lon, sog, and cod have been standardized, i.e. range = [0,1).
            max_seqlen: (optional) max sequence length. Default is
        """

        self.max_seqlen = max_seqlen
        self.device = device

        self.l_data = l_data

    def __len__(self):
        return len(self.l_data)

    def __getitem__(self, idx):
        """Gets items.

        Returns:
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a
            padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        V = self.l_data[idx]
        m_v = V["traj"][:, :2]  # lat, lon, sog, cog前4列
        #         m_v[m_v==1] = 0.9999
        m_v[m_v > 0.9999] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)  # 返回这两个长度中的较小值，序列的实际长度


        seq = np.zeros((self.max_seqlen, 2))
        seq[:seqlen, :] = m_v[:seqlen, :]
        seq = torch.tensor(seq, dtype=torch.float32)  # 转化为torch.tensor

        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.  # 表示这些位置是非padding

        seqlen = torch.tensor(seqlen, dtype=torch.int)

        return seq, mask, seqlen


class AISDataset_grad(Dataset):
    """Customized Pytorch dataset.
    Return the positions and the gradient of the positions.
    """

    def __init__(self,
                 l_data,
                 dlat_max=0.04,
                 dlon_max=0.04,
                 max_seqlen=96,
                 dtype=torch.float32,
                 device=torch.device("cpu")):
        """
        Args
            l_data: list of dictionaries, each element is an AIS trajectory. 
                l_data[idx]["mmsi"]: vessel's MMSI.
                l_data[idx]["traj"]: a matrix whose columns are 
                    [LAT, LON, SOG, COG, TIMESTAMP]
                lat, lon, sog, and cod have been standardized, i.e. range = [0,1).
            dlat_max, dlon_max: the maximum value of the gradient of the positions.
                dlat_max = max(lat[idx+1]-lat[idx]) for all idx.
            max_seqlen: (optional) max sequence length. Default is
        """

        self.dlat_max = dlat_max
        self.dlon_max = dlon_max
        self.dpos_max = np.array([dlat_max, dlon_max])
        self.max_seqlen = max_seqlen
        self.device = device

        self.l_data = l_data

    def __len__(self):
        return len(self.l_data)

    def __getitem__(self, idx):
        """Gets items.
        
        Returns:
            seq: Tensor of (max_seqlen, [lat,lon,sog,cog]).
            mask: Tensor of (max_seqlen, 1). mask[i] = 0.0 if x[i] is a
            padding.
            seqlen: sequence length.
            mmsi: vessel's MMSI.
            time_start: timestamp of the starting time of the trajectory.
        """
        V = self.l_data[idx]
        m_v = V["traj"][:, :4]  # lat, lon, sog, cog
        m_v[m_v == 1] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)
        seq = np.zeros((self.max_seqlen, 4))
        # lat and lon
        seq[:seqlen, :2] = m_v[:seqlen, :2]
        # dlat and dlon
        dpos = (m_v[1:, :2] - m_v[:-1, :2] + self.dpos_max) / (2 * self.dpos_max)
        dpos = np.concatenate((dpos[:1, :], dpos), axis=0)
        dpos[dpos >= 1] = 0.9999
        dpos[dpos <= 0] = 0.0
        seq[:seqlen, 2:] = dpos[:seqlen, :2]

        # convert to Tensor
        seq = torch.tensor(seq, dtype=torch.float32)

        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.

        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi = torch.tensor(V["mmsi"], dtype=torch.int)
        time_start = torch.tensor(V["traj"][0, 4], dtype=torch.int)

        return seq, mask, seqlen, mmsi, time_start
