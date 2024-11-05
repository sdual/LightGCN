import numpy as np
import pandas as pd

from lightgcn.network import init_adj_matrix


def test_init_adj_matrix():
    num_users = 2
    num_items = 3
    user_id_idx = pd.Series([1, 1, 0])
    item_id_idx = pd.Series([0, 1, 2])
    actual = init_adj_matrix(num_users, num_items, user_id_idx, item_id_idx)

    expected = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.70710677, 0.70710677, 0.0],
            [0.0, 0.70710677, 0.0, 0.0, 0.0],
            [0.0, 0.70710677, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    print(type(actual))
    np.testing.assert_array_equal(actual.toarray(), expected)
