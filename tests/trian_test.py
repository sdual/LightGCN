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
        random.seed(42)

        train_df, test_df = create_movielens_dataset(head=10)

        batch_size: int = 10
        epochs: int = 4
        num_layers: int = 2
        num_users: int = len(train_df["user_id_idx"].unique())
        num_items: int = len(train_df["item_id_idx"].unique())
        vec_dim: int = 3
        lr: float = 0.01
        reg_param = 0.5

        model = LightGCN(
            batch_size,
            epochs,
            num_layers,
            num_users,
            num_items,
            vec_dim,
            InitDist.XAVIER_UNIFORM,
            lr,
            reg_param,
        )

        model = model.fit(train_df)
        actual = model._network._init_embed.weight
        expected = torch.Tensor(
            [
                [0.24587622, -0.30305746, -0.5168112],
                [-0.17767076, -0.39160785, 0.42170766],
                [0.12335217, 0.19445957, 0.13921043],
                [-0.01340425, 0.40201655, -0.3612666],
                [-0.00430678, -0.34485203, 0.13466291],
                [-0.15396093, 0.13267091, -0.07757735],
                [0.42781708, -0.29469374, -0.29687238],
                [-0.29677236, 0.4670942, 0.14890906],
                [0.50144184, -0.44473982, -0.52636],
                [-0.4004701, -0.36767477, 0.21333274],
                [0.19897717, 0.503499, -0.32988557],
                [-0.34332564, 0.3243893, -0.19168992],
                [0.30788788, -0.17277794, 0.3578891],
                [-0.47448456, -0.27350226, 0.19481541],
                [0.08272194, -0.10851637, 0.3716828],
                [0.4201203, -0.44673657, -0.33932137],
            ]
        )

        torch.testing.assert_close(actual, expected)
