import random

import pandas as pd
import torch

from lightgcn.embedding import InitDist
from lightgcn.train import LightGCN, _PosNegItemSelector
from tests.helper import create_movielens_dataset


class TestPosNegItemSelector:
    def test_select_pos_neg_item_idxs(self):
        # Fix the seed to use the same items in the test.
        random.seed(42)
        user_items = {
            "user_id_idx": {0: 0, 1: 1, 2: 2, 3: 3, 4: 4},
            "item_id_idx_list": {
                0: [
                    37,
                    112,
                    206,
                ],
                1: [
                    304,
                    283,
                ],
                3: [
                    326,
                    359,
                    270,
                ],
                4: [
                    180,
                    20,
                    23,
                ],
            },
        }
        grouped_by_user_df = pd.DataFrame(user_items)
        user_id_idxs = [0, 3]
        unique_item_idxs = [37, 112, 206, 304, 284, 326, 359, 270, 180, 20, 23]

        actual = _PosNegItemSelector.select_pos_neg_item_idxs(
            user_id_idxs, unique_item_idxs, grouped_by_user_df
        )

        expected = pd.DataFrame(
            {
                "user_id_idx": [0, 3],
                "sampled_pos_item_idx": [206, 326],
                "sampled_neg_item_idx": [326, 180],
            }
        )
        pd.testing.assert_frame_equal(actual, expected)


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

    def test_extract_pos_neg_item_idxs():
        # Fix the seed for test to use the same items in _extract_pos_neg_item_idxs method.
        random.seed(42)

        batch_size: int = 10
        epochs: int = 4
        num_users: int = 3
        num_items: int = 5
        vec_dim: int = 3
        lr: float = 0.01

        model = LightGCN(
            batch_size, epochs, num_users, num_items, vec_dim, InitDist.XAVIER_UNIFORM, lr
        )

        model._extract_pos_neg_item_idxs()
