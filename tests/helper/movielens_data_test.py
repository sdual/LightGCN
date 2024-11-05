from tests.helper.movielens_data import create_movielens_dataset


def test_create_movielens_dataset():
    train_df, test_df = create_movielens_dataset()
    print(train_df.head())
    print(test_df.head())
