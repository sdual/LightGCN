from typing import Protocol

import torch.nn as nn


class EmbeddingInitializer(Protocol):
    def init_embedding(self) -> nn.Embedding: ...

    def num_users(self) -> int: ...

    def num_items(self) -> int: ...


class XavierUniformInitializer:
    def __init__(self, num_users: int, num_items: int, vector_dim: int):
        self._num_users: int = num_users
        self._num_items: int = num_items
        self._vector_dim: int = vector_dim

    def init_embedding(self) -> nn.Embedding:
        num_embedding_vectors: int = self._num_items + self._num_users
        embed = nn.Embedding(
            num_embedding_vectors,
            self._vector_dim,
        )
        nn.init.xavier_uniform_(embed.weight)
        return embed

    def num_users(self) -> int:
        return self._num_users

    def num_items(self) -> int:
        return self._num_items
