import torch.nn as nn

from lightgcn.embedding import EmbeddingInitializer


class LightGCN(nn.Module):
    def __init__(self, embed_initializer: EmbeddingInitializer):
        self._embed: nn.Embedding = embed_initializer.create_init_embedding()

    # This method creates D^{-1/2} A D^{-1/2} in https://arxiv.org/abs/2002.02126
    def _init_adj_matrix(self):
        rating_matrix = None

    def forward(self):
        pass
