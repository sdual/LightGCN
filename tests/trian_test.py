import torch

from lightgcn.embedding import InitDist
from lightgcn.train import LightGCN
from tests.helper import create_movielens_dataset


class TestLightGCN:
    def test_fit(self):
        torch.manual_seed(42)

        batch_size: int = 10
        epochs: int = 4
        num_users: int = 3
        num_items: int = 5
        vec_dim: int = 3
        lr: float = 0.01

        model = LightGCN(
            batch_size, epochs, num_users, num_items, vec_dim, InitDist.XAVIER_UNIFORM, lr
        )

        train_df, test_df = create_movielens_dataset(head=10)
        cols = ["user_id_idx", "item_id_idx", "rating"]
        print(train_df[cols])
