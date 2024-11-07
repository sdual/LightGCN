from enum import Enum

import torch.nn as nn


class InitDist(Enum):
    XAVIER_UNIFORM = 1
    XAVIER_NORMAL = 2


class EmbeddingLayer:
    def __init__(self, num_users: int, num_items: int, vector_dim: int, init_dist: InitDist):
        self._num_users: int = num_users
        self._num_items: int = num_items
        self._vector_dim: int = vector_dim
        self._init_dist: InitDist = init_dist

    def init_embedding(self) -> nn.Embedding:
        num_embedding_vectors: int = self._num_items + self._num_users
        embed = nn.Embedding(
            num_embedding_vectors,
            self._vector_dim,
        )
        self._initialize_params(embed)
        return embed

    def _initialize_params(self, embed: nn.Embedding):
        if self._init_dist == InitDist.XAVIER_UNIFORM:
            nn.init.xavier_uniform_(embed.weight)
        elif self._init_dist == InitDist.XAVIER_NORMAL:
            nn.init.xavier_normal_(embed.weight)
        else:
            raise ValueError(f"Unsuported initial distribution: {self._init_dist}")
