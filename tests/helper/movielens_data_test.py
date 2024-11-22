import os

from lightgcn import PROJECT_ROOT
from tests.helper.movielens_data import create_movielens_dataset


def test_create_movielens_dataset():
    file_path = os.path.join(PROJECT_ROOT, "tests/helper/sample_data.tsv")
    train_df, test_df = create_movielens_dataset(file_path=file_path)
    print(train_df.head())
    print(test_df.head())
