import random
from typing import Any, Self

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lightgcn.columns import FeatureCol
from lightgcn.dataset import RecommendUserDataset
from lightgcn.embedding import InitDist
from lightgcn.network import LightGCNNetwork


class LightGCN:
    _ITEM_ID_IDX_LIST_COL: str = "item_id_idx_list"
    _SAMPLED_POS_ITEM_IDX_COL: str = "sampled_pos_item_idx"
    _SAMPLED_NEG_ITEM_IDX_COL: str = "sampled_neg_item_idx"

    def __init__(
        self,
        batch_size: int,
        epochs: int,
        num_users: int,
        num_items: int,
        vec_dim: int,
        init_dict: InitDist,
        lr: float,
    ):
        self._batch_size: int = batch_size
        self._epochs: int = epochs
        self._num_users: int = num_users
        self._num_items: int = num_items
        self._vec_dim: int = vec_dim
        self._init_dict: InitDist = init_dict
        self._network: LightGCNNetwork | None = None
        self._lr: float = lr

    # ratings contains user_id_idx, item_id_idx and its ratings.
    def fit(self, rating_df: pd.DataFrame) -> Self:
        user_idx_items_df = self._groupby_user_id_idx(rating_df)
        self._network = LightGCNNetwork(
            self._num_users, self._num_items, self._vec_dim, self._init_dict
        )
        unique_item_idxs = rating_df[FeatureCol.ITEM_ID_IDX].unique()
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self._lr)

        self._network.train()
        unique_user_id_idxs = user_idx_items_df[FeatureCol.USER_ID_IDX].values
        dataset = RecommendUserDataset(unique_user_id_idxs)
        dataloader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        for epoch in range(self._epochs):
            for user_id_idx in dataloader:
                self._network()

        return self

    def predict(self) -> np.ndarray:
        if self._network is None:
            raise RuntimeError("LightGCN is not trained")
        self._network.eval()
        pass

    def _groupby_user_id_idx(self, rating_df: pd.DataFrame) -> pd.DataFrame:
        item_grouped_df = pd.DataFrame(
            rating_df.groupby(FeatureCol.USER_ID_IDX, as_index=False)[FeatureCol.ITEM_ID_IDX].apply(
                lambda x: list(x)
            )
        ).rename(columns={FeatureCol.ITEM_ID_IDX: self._ITEM_ID_IDX_LIST_COL})
        return item_grouped_df

    def _extract_pos_neg_item_idxs(
        self,
        user_id_idxs: list[int],
        unique_item_idxs: np.ndarray,
        grouped_by_user_df: pd.DataFrame,
    ) -> dict[str, np.ndarray]:
        sampled_pos_item_idx = grouped_by_user_df[
            grouped_by_user_df[FeatureCol.USER_ID_IDX].isin(user_id_idxs)
        ][self._ITEM_ID_IDX_LIST_COL].apply(lambda item_idxs: random.choice(item_idxs))

        grouped_by_user_df[self._SAMPLED_POS_ITEM_IDX_COL] = sampled_pos_item_idx

        sampled_neg_item_idx = grouped_by_user_df[
            grouped_by_user_df[FeatureCol.USER_ID_IDX].isin(user_id_idxs)
        ][self._ITEM_ID_IDX_LIST_COL].apply(
            lambda x: random.choice(list(set(unique_item_idxs) - set(x)))
        )
        grouped_by_user_df[self._SAMPLED_NEG_ITEM_IDX_COL] = sampled_neg_item_idx

        return grouped_by_user_df[
            [FeatureCol.USER_ID_IDX, self._SAMPLED_POS_ITEM_IDX_COL, self._SAMPLED_NEG_ITEM_IDX_COL]
        ]
