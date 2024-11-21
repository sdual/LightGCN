from torch.utils.data import DataLoader

from lightgcn.dataset import RecommendUserDataset
from tests.helper.movielens_data import create_movielens_dataset


class TestRecommendDataset:
    def test_dataloader(self):
        train_df, _ = create_movielens_dataset(head=10)
        dataset = RecommendUserDataset(train_df)
        dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
        for user_id_idx, item_id_idx in dataloader:
            print(user_id_idx)
            print(item_id_idx)
