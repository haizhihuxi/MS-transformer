

import numpy as np


import torch
from torch.utils.data import Dataset


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
