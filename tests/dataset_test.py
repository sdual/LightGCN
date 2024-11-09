import numpy as np
from torch.utils.data import DataLoader

from lightgcn.dataset import RecommendUserDataset
from tests.helper.movielens_data import create_movielens_dataset


class TestRecommendDataset:
    def test_dataloader(self):
        train_df, _ = create_movielens_dataset()
        unique_user_ids: np.ndarray = train_df["user_id"].unique()
        dataset = RecommendUserDataset(unique_user_ids)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
        for user_id_idx in dataloader:
            print(user_id_idx)
