import torch

from lightgcn.embedding import EmbeddingLayer, InitDist


class TestXavierUniformEmbedding:
    def test_init_embedding(self):
        num_users: int = 3
        num_items: int = 5
        vector_dim: int = 10
        embed_initer = EmbeddingLayer(num_users, num_items, vector_dim, InitDist.XAVIER_UNIFORM)
        actual = embed_initer.init_embedding()
        assert actual.weight.shape == torch.Size([8, 10])
