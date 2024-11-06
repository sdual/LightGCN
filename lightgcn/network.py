import numpy as np
import pandas as pd
import torch.nn as nn
from scipy.sparse import csr_matrix, diags, dok_matrix

from lightgcn.embedding import EmbeddingInitializer


def _take_power_1over2_if_non_zero(value: np.float32) -> np.float32:
    if value == np.float32(0.0):
        return np.float32(0.0)
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
    def __init__(self, embed_initializer: EmbeddingInitializer, ratings: pd.DataFrame):
        self._embed: nn.Embedding = embed_initializer.init_embedding()
        self._num_users: int = embed_initializer.num_users()
        self._num_items: int = embed_initializer.num_items()
        self._ratings: pd.DataFrame = ratings

    def forward(self):
        pass
