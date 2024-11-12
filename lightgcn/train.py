from typing import Self

import numpy as np
import pandas as pd
import torch

from lightgcn.columns import FeatureCol
from lightgcn.embedding import InitDist
from lightgcn.network import LightGCNNetwork


class LightGCN:
    _ITEM_ID_IDX_LIST_COL: str = "item_id_idx_list"

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
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self._lr)
        self._network.train()

        for epoch in range(self._epochs):
            pass
        pass

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
