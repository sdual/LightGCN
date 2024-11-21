import random
from typing import Self

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from lightgcn.columns import FeatureCol
from lightgcn.dataset import RecommendUserDataset
from lightgcn.embedding import InitDist
from lightgcn.loss import bpr_loss
from lightgcn.network import LightGCNNetwork


class _TempCol:
    ITEM_ID_IDX_LIST_COL: str = "item_id_idx_list"
    SAMPLED_POS_ITEM_IDX_COL: str = "sampled_pos_item_idx"
    SAMPLED_NEG_ITEM_IDX_COL: str = "sampled_neg_item_idx"


class _PosNegItemSelector:
    @classmethod
    def select_pos_neg_item_idxs(
        cls,
        user_id_idxs: torch.Tensor,
        unique_item_idxs: np.ndarray,
        grouped_by_user_df: pd.DataFrame,
    ) -> pd.DataFrame:
        # TODO: Select cpu or gpu according to the environment.
        user_id_idxs_array = user_id_idxs.to("cpu").detach().numpy()
        target_user_df = (
            pd.DataFrame({FeatureCol.USER_ID_IDX: user_id_idxs_array})
            .reset_index(drop=True)
            .astype(np.int64)
        )

        target_user_df[_TempCol.SAMPLED_NEG_ITEM_IDX_COL] = cls._extract_neg_item_idxs(
            target_user_df, unique_item_idxs, grouped_by_user_df
        )
        return target_user_df

    @classmethod
    def _extract_neg_item_idxs(
        cls,
        target_user_df: pd.DataFrame,
        unique_item_idxs: np.ndarray,
        grouped_by_user_df: pd.DataFrame,
    ) -> pd.Series:
        return target_user_df.apply(
            lambda x: random.choice(
                list(
                    set(unique_item_idxs)
                    - set(cls._extract_item_idxs(x[FeatureCol.USER_ID_IDX], grouped_by_user_df))
                )
            ),
            axis=1,
        ).astype(np.int64)

    @classmethod
    def _extract_item_idxs(cls, user_id_idx: int, grouped_by_user_df: pd.DataFrame) -> list[int]:
        grouped_by_user_df.reset_index(drop=True, inplace=True)
        return grouped_by_user_df[
            grouped_by_user_df[FeatureCol.USER_ID_IDX] == user_id_idx
        ].reset_index(drop=True)[_TempCol.ITEM_ID_IDX_LIST_COL][0]


class LightGCN:
    def __init__(
        self,
        batch_size: int,
        epochs: int,
        num_layers: int,
        num_users: int,
        num_items: int,
        vec_dim: int,
        init_dict: InitDist,
        lr: float,
        reg_param: float,
    ):
        self._batch_size: int = batch_size
        self._epochs: int = epochs
        self._num_layers: int = num_layers
        self._num_users: int = num_users
        self._num_items: int = num_items
        self._vec_dim: int = vec_dim
        self._init_dict: InitDist = init_dict
        self._network: LightGCNNetwork | None = None
        self._lr: float = lr
        self._reg_param: float = reg_param
        self._train_loss_history: list[float] = []
        self._val_loss_history: list[float] = []
        self._val_user_id_idxs: torch.Tensor | None = None
        self._val_unique_item_idxs: np.ndarray | None = None

    # ratings contains user_id_idx, item_id_idx and its ratings.
    def fit(self, rating_df: pd.DataFrame, val_rating_df: pd.DataFrame | None = None) -> Self:
        self._network = LightGCNNetwork(
            self._num_users,
            self._num_items,
            self._vec_dim,
            self._num_layers,
            self._init_dict,
            rating_df[[FeatureCol.USER_ID_IDX, FeatureCol.ITEM_ID_IDX]],
        )
        unique_item_idxs = rating_df[FeatureCol.ITEM_ID_IDX].unique()
        optimizer = torch.optim.Adam(self._network.parameters(), lr=self._lr)

        dataset = RecommendUserDataset(rating_df)
        dataloader = DataLoader(dataset, batch_size=self._batch_size, shuffle=True)

        self._network.train()
        for epoch in range(self._epochs):
            # TODO: Use logger.
            print(f"epoch: {epoch}")
            for user_id_idxs, pos_item_id_idx in dataloader:
                optimizer.zero_grad()
                neg_item_idx_tensor = self._extract_neg_items(
                    user_id_idxs, unique_item_idxs, rating_df
                )

                embeddings = self._network(user_id_idxs, pos_item_id_idx, neg_item_idx_tensor)
                loss = bpr_loss(
                    embeddings["user_emb"],
                    embeddings["pos_item_emb"],
                    embeddings["neg_item_emb"],
                    embeddings["user_0emb"],
                    embeddings["pos_item_0emb"],
                    embeddings["neg_item_0emb"],
                    self._reg_param,
                )

                loss.backward()
                optimizer.step()
                self._train_loss_history.append(loss.item())

                # TODO: User logger.
                print(f"train loss: {loss.item()}")
            if val_rating_df is not None:
                self._calc_val_loss(rating_df)
        return self

    def _extract_neg_items(
        self,
        user_id_idx: torch.Tensor,
        unique_item_idxs: np.ndarray,
        rating_df: pd.DataFrame,
    ) -> dict[str, torch.Tensor]:
        user_idx_items_df = self._groupby_user_id_idx(rating_df)
        user_sampled_item_idxs = _PosNegItemSelector.select_pos_neg_item_idxs(
            user_id_idx, unique_item_idxs, user_idx_items_df
        )
        neg_item_idx_tensor = torch.from_numpy(
            user_sampled_item_idxs[_TempCol.SAMPLED_NEG_ITEM_IDX_COL].values.astype(np.long)
        )
        return neg_item_idx_tensor

    def predict(self, user_item_df: pd.DataFrame) -> np.ndarray:
        if self._network is None:
            raise RuntimeError("LightGCN is not trained")
        self._network.eval()

    def _calc_val_loss(self, val_rating_df: pd.DataFrame):
        if self._val_user_id_idxs is None:
            self._val_user_id_idxs = torch.from_numpy(
                val_rating_df[FeatureCol.USER_ID_IDX].values.astype(np.long)
            )

        if self._val_unique_item_idxs is None:
            self._val_unique_item_idxs = np.unique(val_rating_df[FeatureCol.ITEM_ID_IDX].values)
        neg_item_idx_tensor = self._extract_neg_items(
            self._val_user_id_idxs, self._val_unique_item_idxs, val_rating_df
        )
        pos_item_id_idx = torch.from_numpy(
            val_rating_df[FeatureCol.ITEM_ID_IDX].values.astype(np.long)
        )

        self._network.eval()
        embeddings = self._network(self._val_user_id_idxs, pos_item_id_idx, neg_item_idx_tensor)
        loss = bpr_loss(
            embeddings["user_emb"],
            embeddings["pos_item_emb"],
            embeddings["neg_item_emb"],
            embeddings["user_0emb"],
            embeddings["pos_item_0emb"],
            embeddings["neg_item_0emb"],
            self._reg_param,
        )
        self._val_loss_history.append(loss)

    def _groupby_user_id_idx(self, rating_df: pd.DataFrame) -> pd.DataFrame:
        item_grouped_df = pd.DataFrame(
            rating_df.groupby(FeatureCol.USER_ID_IDX, as_index=False)[FeatureCol.ITEM_ID_IDX].apply(
                lambda x: list(x)
            )
        ).rename(columns={FeatureCol.ITEM_ID_IDX: _TempCol.ITEM_ID_IDX_LIST_COL})
        return item_grouped_df

    def train_loss_history(self) -> list[float]:
        return self._train_loss_history

    def val_loss_history(self) -> list[float]:
        return self._val_loss_history
