import numpy as np
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
    np.testing.assert_array_equal(actual.toarray(), expected)


class TestLightGCN:
    def test_forward(self):
        num_users = 2
        num_items = 3
        vec_dim = 10
        num_layers = 3
        init_dist = InitDist.XAVIER_UNIFORM

        model = LightGCN(num_users, num_items, vec_dim, num_layers, init_dist)
        actual = model(torch.Tensor([0, 1]))
        print(actual)
