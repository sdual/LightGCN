import numpy as np
import pandas as pd
import torch

from lightgcn.embedding import InitDist
from lightgcn.network import LightGCN, init_adj_matrix


def test_init_adj_matrix():
    num_users = 2
    num_items = 3
    user_id_idx = np.array([1, 1, 0])
    item_id_idx = np.array([0, 1, 2])
    actual = init_adj_matrix(num_users, num_items, user_id_idx, item_id_idx)

    # The expected matrix is D^{-1/2} A D^{-1/2} in the article https://arxiv.org/abs/2002.02126
    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.70710677, 0.70710677, 0.0],
            [0.0, 0.70710677, 0.0, 0.0, 0.0],
            [0.0, 0.70710677, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    indices = np.array([[0, 1, 1, 2, 3, 4], [4, 2, 3, 1, 1, 0]])
    expected_adj_norm_data = np.array([1.0, 0.70710677, 0.70710677, 0.70710677, 0.70710677, 1.0])
    expected_shape = (5, 5)
    expected = torch.sparse_coo_tensor(
        torch.LongTensor(indices),
        torch.FloatTensor(expected_adj_norm_data),
        torch.Size(expected_shape),
    )
    torch.testing.assert_close(actual, expected)


class TestLightGCN:
    def test_forward(self):
        torch.manual_seed(42)

        num_users = 2
        num_items = 3
        vec_dim = 10
        num_layers = 3
        init_dist = InitDist.XAVIER_UNIFORM
        user_item_id_idxs = pd.DataFrame(
            {
                "user_id_idx": [0, 1],
                "item_id_idx": [2, 1],
            }
        )

        model = LightGCN(num_users, num_items, vec_dim, num_layers, init_dist, user_item_id_idxs)
        user_id_idxs = torch.Tensor([0, 1])
        actual_user_emb, actual_item_emb = model(user_id_idxs)
        expected_user_emb = torch.Tensor(
            [
                [
                    -0.00600985,
                    -0.36156362,
                    -0.46536398,
                    -0.38996324,
                    0.6005329,
                    0.16511077,
                    0.40173903,
                    -0.55591595,
                    -0.5213489,
                    -0.3526634,
                ],
                [
                    0.07683477,
                    0.02137942,
                    0.0011256,
                    -0.04321894,
                    -0.34440225,
                    -0.13658133,
                    0.12604363,
                    -0.35736597,
                    0.19934382,
                    -0.29105037,
                ],
            ]
        )

        expected_item_emb = torch.Tensor(
            [
                [
                    0.0904484,
                    -0.12284943,
                    -0.07979213,
                    0.0482052,
                    0.03342647,
                    -0.04031252,
                    0.09424686,
                    0.10748728,
                    -0.11465997,
                    -0.08441197,
                ],
                [
                    0.07683477,
                    0.02137942,
                    0.0011256,
                    -0.04321894,
                    -0.34440225,
                    -0.13658135,
                    0.12604363,
                    -0.35736597,
                    0.19934383,
                    -0.29105037,
                ],
                [
                    -0.00600985,
                    -0.36156362,
                    -0.46536398,
                    -0.38996324,
                    0.6005329,
                    0.16511077,
                    0.40173903,
                    -0.55591595,
                    -0.5213489,
                    -0.3526634,
                ],
            ]
        )
        torch.testing.assert_close(actual_user_emb, expected_user_emb)
        torch.testing.assert_close(actual_item_emb, expected_item_emb)
