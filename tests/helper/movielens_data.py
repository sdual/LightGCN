import os

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from lightgcn import PROJECT_ROOT
from lightgcn.columns import FeatureCol


def create_movielens_dataset(head: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    file_path = os.path.join(PROJECT_ROOT, "testdata/ml-100k/u.data")
    movielens_df = pd.read_csv(
        file_path,
        sep="\t",
        names=[
            FeatureCol.USER_ID,
            FeatureCol.ITEM_ID,
            FeatureCol.RATING,
            "timestamp",
        ],
    )

    movielens_df = movielens_df[movielens_df[FeatureCol.RATING] >= 3]
    if head is not None:
        movielens_df = movielens_df[:head]

    train_df, test_df = train_test_split(movielens_df.values, test_size=0.2, random_state=42)
    train_df = pd.DataFrame(train_df, columns=movielens_df.columns)
    test_df = pd.DataFrame(test_df, columns=movielens_df.columns)

    le_user = preprocessing.LabelEncoder()
    le_item = preprocessing.LabelEncoder()
    train_df[FeatureCol.USER_ID_IDX] = le_user.fit_transform(train_df[FeatureCol.USER_ID].values)
    train_df[FeatureCol.ITEM_ID_IDX] = le_item.fit_transform(train_df[FeatureCol.ITEM_ID].values)

    train_user_ids = train_df[FeatureCol.USER_ID].unique()
    train_item_ids = train_df[FeatureCol.ITEM_ID].unique()

    test_df = test_df[
        (test_df[FeatureCol.USER_ID].isin(train_user_ids))
        & (test_df[FeatureCol.ITEM_ID].isin(train_item_ids))
    ]

    test_df[FeatureCol.USER_ID_IDX] = le_user.transform(test_df[FeatureCol.USER_ID].values)
    test_df[FeatureCol.ITEM_ID_IDX] = le_item.transform(test_df[FeatureCol.ITEM_ID].values)
    return train_df, test_df
