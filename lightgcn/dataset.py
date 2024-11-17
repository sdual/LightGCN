import numpy as np
import torch
from torch.utils.data import Dataset


class RecommendUserDataset(Dataset):
    def __init__(self, unique_user_id_idxs: np.ndarray):
        self._unique_user_id_idxs: np.ndarray = unique_user_id_idxs

    def __len__(self):
        return len(self._unique_user_id_idxs)

    def __getitem__(self, index):
        user_id_idx = self._unique_user_id_idxs[index]
        # the data type of user_id is integer.
        return torch.tensor(user_id_idx).long()
