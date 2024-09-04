
"""Customized Pytorch Dataset.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class AISDataset(Dataset):
    """Customized Pytorch dataset.
    """

    def __init__(self,
                 l_data,
                 max_seqlen=96,
                 device=torch.device("cpu")):

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
        m_v = V["traj"][: ,:4]
        m_v[m_v > 0.9999] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)

        target = np.full((1 , 2), np.nan)
        target[:, 0] = m_v[seqlen - 1][0]
        target[:, 1] = m_v[seqlen - 1][1]


        seq = np.zeros((self.max_seqlen, 4))
        seq[:seqlen, :] = m_v[:seqlen, :]
        seq = torch.tensor(seq, dtype=torch.float32)

        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.

        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi = torch.tensor(V["mmsi"], dtype=torch.int)
        time_start = torch.tensor(V["traj"][0, 4], dtype=torch.int)

        return seq, target, mask, seqlen, mmsi, time_start


class AISDataset_train(Dataset):
    """Customized Pytorch dataset.
    """

    def __init__(self,
                 l_data,
                 max_seqlen=96,
                 device=torch.device("cpu")):

        self.max_seqlen = max_seqlen
        self.device = device

        self.l_data = l_data

    def __len__(self):
        return len(self.l_data)

    def __getitem__(self, idx):
        V = self.l_data[idx]
        m_v = V["traj"][:, :4]
        weight_v = V["traj"][:, 6:]
        m_v[m_v > 0.9999] = 0.9999
        seqlen = min(len(m_v), self.max_seqlen)

        target = np.full((1, 2), np.nan)
        target[:, 0] = m_v[seqlen - 1][0]
        target[:, 1] = m_v[seqlen - 1][1]

        seq = np.zeros((self.max_seqlen, 4))
        seq[:seqlen, :] = m_v[:seqlen, :]
        seq = torch.tensor(seq, dtype=torch.float32)

        seq_weight = np.zeros((self.max_seqlen, 5))
        seq_weight[:seqlen, :] = weight_v[:seqlen, :]

        last_non_zero_weight = seq_weight[seqlen - 1]
        if seqlen < self.max_seqlen:
            for i in range(seqlen, self.max_seqlen):
                seq_weight[i] = last_non_zero_weight
        seq_weight = torch.tensor(seq_weight, dtype=torch.float32)

        mask = torch.zeros(self.max_seqlen)
        mask[:seqlen] = 1.

        seqlen = torch.tensor(seqlen, dtype=torch.int)
        mmsi = torch.tensor(V["mmsi"], dtype=torch.int)

        return seq, target, mask, seqlen, mmsi, seq_weight
