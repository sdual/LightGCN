import random

import matplotlib.pyplot as plt
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
        user_id_idxs = torch.Tensor([0, 3])
        unique_item_idxs = [37, 112, 206, 304, 284, 326, 359, 270, 180, 20, 23]

        actual = _PosNegItemSelector.select_pos_neg_item_idxs(
            user_id_idxs, unique_item_idxs, grouped_by_user_df
        )

        expected = pd.DataFrame(
            {
                "user_id_idx": [0, 3],
                "sampled_neg_item_idx": [359, 37],
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
                [0.24542502, -0.36240292, -0.5168443],
                [-0.24818495, -0.39210823, 0.4246173],
                [0.12658532, 0.27227414, 0.18087305],
                [0.02924936, 0.4017512, -0.41193542],
                [-0.00383235, -0.34490326, 0.17962709],
                [-0.15543348, 0.13560265, -0.15516637],
                [0.4333162, -0.29340374, -0.29733005],
                [-0.2964189, 0.46628445, 0.19561031],
                [0.50229675, -0.4247789, -0.51932096],
                [-0.40400985, -0.3446185, 0.18954942],
                [0.1646341, 0.43039107, -0.25345102],
                [-0.3530697, 0.26010197, -0.19100085],
                [0.3021461, -0.09391501, 0.2847341],
                [-0.39908442, -0.2509169, 0.15548828],
                [0.1106717, -0.10961639, 0.29698443],
                [0.34234723, -0.3684882, -0.26969254],
            ]
        )
        torch.testing.assert_close(actual, expected)

    def test_train_all_data(self):
        torch.manual_seed(42)
        random.seed(42)

        train_df, test_df = create_movielens_dataset()

        batch_size: int = 1024
        epochs: int = 30
        num_layers: int = 3
        num_users: int = len(train_df["user_id_idx"].unique())
        num_items: int = len(train_df["item_id_idx"].unique())
        vec_dim: int = 64
        lr: float = 0.005
        reg_param = 0.0001

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
        train_loss_history = model.train_loss_history()
        val_loss_history = model.val_loss_history()
        plt.plot(train_loss_history, label="training loss")
        plt.plot(val_loss_history, label="validation loss")
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.legend()

        plt.savefig("train-loss.png", format="png", dpi=100)
