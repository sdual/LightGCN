import pandas as pd
import torch
from torch.utils.data import Dataset


class RecommendUserDataset(Dataset):
    def __init__(self, rating_df: pd.DataFrame):
        self._rating_df: pd.DataFrame = rating_df

    def __len__(self):
        return len(self._rating_df)

    def __getitem__(self, index):
        user_id_idx = self._rating_df["user_id_idx"][index]
        item_id_idx = self._rating_df["item_id_idx"][index]
        return torch.tensor(user_id_idx).long(), torch.tensor(item_id_idx).long()
