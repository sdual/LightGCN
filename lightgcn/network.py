import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import csr_matrix, diags, dok_matrix

from lightgcn.embedding import EmbeddingLayer, InitDist


def _take_power_1over2_if_non_zero(value: np.float32) -> np.float32:
    if value == np.float32(0.0):
        return value
    else:
        return np.float32(np.power(value, -0.5))


_take_power_1over2_if_non_zero_func = np.frompyfunc(_take_power_1over2_if_non_zero, 1, 1)


# This method creates D^{-1/2} A D^{-1/2} in the article https://arxiv.org/abs/2002.02126
def init_adj_matrix(
    num_users: int, num_items: int, user_id_idx: np.ndarray, item_id_idx: np.ndarray
) -> csr_matrix:
    rating_matrix = dok_matrix((num_users, num_items), dtype=np.float32)
    rating_matrix[user_id_idx, item_id_idx] = 1.0

    adj_matrix_dim = num_users + num_items
    adj_matrix = dok_matrix((adj_matrix_dim, adj_matrix_dim), dtype=np.float32)

    adj_matrix = adj_matrix.tolil()
    rating_matrix = rating_matrix.tolil()

    adj_matrix[:num_users, num_users:] = rating_matrix
    adj_matrix[num_users:, :num_users] = rating_matrix.T

    # adj_matrix is the matrix A.
    adj_matrix = adj_matrix.todok()
    row_sums = np.array(adj_matrix.sum(1))
    d_pow_1over2_diags = _take_power_1over2_if_non_zero_func(row_sums.flatten()).astype(np.float32)

    d_pow_1over2 = diags(d_pow_1over2_diags)
    results = d_pow_1over2.dot(adj_matrix).dot(d_pow_1over2)
    return results


class LightGCN(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        vec_dim: int,
        num_layers: int,
        init_dist: InitDist,
    ):
        super(LightGCN, self).__init__()
        embed_initializer = EmbeddingLayer(num_users, num_items, vec_dim, init_dist)
        self._embed: nn.Embedding = embed_initializer.init_embedding()
        self._num_users: int = num_users
        self._num_items: int = num_items
        self._norm_adj: csr_matrix = init_adj_matrix(num_items, num_items)
        self._num_layers: int = num_layers

    def forward(self, user_idxs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embed_weights = self._embed.weight
        all_layer_embed_list = [self._embed.weight] + [
            torch.sparse.mm(self._norm_adj, embed_weights) for _ in range(self._num_layers)
        ]
        all_embed_tensor = torch.stack(all_layer_embed_list)
        mean_all_embed_tensor = torch.mean(all_embed_tensor, dim=0)
        user_embed, item_embed = torch.split(
            mean_all_embed_tensor, [self._num_users, self._num_items]
        )
        return user_embed, item_embed
