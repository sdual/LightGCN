import torch

from lightgcn.embedding import XavierUniformInitializer


class TestXavierUniformInitializer:
    def test_init_embedding(self):
        num_users: int = 3
        num_items: int = 5
        vector_dim: int = 10
        embed_initer = XavierUniformInitializer(num_users, num_items, vector_dim)
        actual = embed_initer.init_embedding()
        assert actual.weight.shape == torch.Size([8, 10])
