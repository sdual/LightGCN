import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix, csr_matrix, diags, dok_matrix

from lightgcn.columns import FeatureCol
from lightgcn.embedding import EmbeddingLayer, InitDist


def _take_power_1over2_if_non_zero(value: np.float32) -> np.float32:
    if value == np.float32(0.0):
        return value
    else:
        return np.float32(np.power(value, -0.5))


_take_power_1over2_if_non_zero_func = np.frompyfunc(_take_power_1over2_if_non_zero, 1, 1)


def init_adj_matrix(
    num_users: int, num_items: int, user_id_idxs: np.ndarray, item_id_idxs: np.ndarray
) -> torch.Tensor:
    rating_matrix = dok_matrix((num_users, num_items), dtype=np.float32)
    rating_matrix[user_id_idxs, item_id_idxs] = 1.0

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
    results: csr_matrix = d_pow_1over2.dot(adj_matrix).dot(d_pow_1over2)
    return _to_space_tensor(results)


def _to_space_tensor(adj_norm_mat: csr_matrix) -> torch.Tensor:
    adj_norm_coo: coo_matrix = adj_norm_mat.tocoo().astype(np.float32)
    indices = np.vstack((adj_norm_coo.row, adj_norm_coo.col))
    adj_norm_tensor = torch.sparse_coo_tensor(
        torch.LongTensor(indices),
        torch.FloatTensor(adj_norm_coo.data),
        torch.Size(adj_norm_coo.shape),
    )
    return adj_norm_tensor


class LightGCNNetwork(nn.Module):
    def __init__(
        self,
        num_users: int,
        num_items: int,
        vec_dim: int,
        num_layers: int,
        init_dist: InitDist,
        pos_interact_user_item_idxs: pd.DataFrame,
    ):
        super(LightGCNNetwork, self).__init__()
        embed_initializer = EmbeddingLayer(num_users, num_items, vec_dim, init_dist)
        self._init_embed: nn.Embedding = embed_initializer.init_embedding()
        self._num_users: int = num_users
        self._num_items: int = num_items
        user_id_idxs = pos_interact_user_item_idxs[FeatureCol.USER_ID_IDX].values
        item_id_idxs = pos_interact_user_item_idxs[FeatureCol.ITEM_ID_IDX].values
        self._norm_adj: torch.Tensor = init_adj_matrix(
            num_users, num_items, user_id_idxs, item_id_idxs
        )
        self._num_layers: int = num_layers

    def forward(
        self, user_idxs: list[int], pos_item_idxs: list[int], neg_item_idxs: list[int]
    ) -> dict[str, torch.Tensor]:
        embed_weights = self._init_embed.weight
        all_layer_embed_list = [embed_weights]
        # Create a list of embeddings [E^{(0)}, E^{(1)}, ..., E^{(K)}].
        for _ in range(self._num_layers):
            embed_weights = torch.sparse.mm(self._norm_adj, embed_weights)
            all_layer_embed_list.append(embed_weights)

        all_embed_tensor = torch.stack(all_layer_embed_list)
        # Average of all the embeddings E = α_0 E^{(0)} + α_1 E^{(1)} + ... + α_K E^{(K)}
        mean_all_embed_tensor = torch.mean(all_embed_tensor, dim=0)
        all_user_embeds, all_item_embeds = torch.split(
            mean_all_embed_tensor, [self._num_users, self._num_items]
        )
        user_emb = all_user_embeds[user_idxs]
        pos_item_emb = all_item_embeds[pos_item_idxs]
        neg_item_emb = all_item_embeds[neg_item_idxs]

        all_init_user_emb, init_item_emb = torch.split(
            self._init_embed.weight, [self._num_users, self._num_items]
        )
        init_user_0emb = all_init_user_emb[user_idxs]
        init_pos_item_0emb = all_init_user_emb[pos_item_idxs]
        init_neg_item_0emb = all_init_user_emb[neg_item_idxs]

        return {
            "user_emb": user_emb,
            "pos_item_emb": pos_item_emb,
            "neg_item_emb": neg_item_emb,
            "user_0emb": init_user_0emb,
            "pos_item_0emb": init_pos_item_0emb,
            "neg_item_0emb": init_neg_item_0emb,
        }
