import os

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from lightgcn import PROJECT_ROOT


def create_movielens_dataset() -> tuple[pd.DataFrame, pd.DataFrame]:
    file_path = os.path.join(PROJECT_ROOT, "testdata/ml-100k/u.data")
    movielens_df = pd.read_csv(
        file_path, sep="\t", names=["user_id", "item_id", "rating", "timestamp"]
    )

    movielens_df = movielens_df[movielens_df["rating"] >= 3]

    train_df, test_df = train_test_split(movielens_df.values, test_size=0.2, random_state=42)
    train_df = pd.DataFrame(train_df, columns=movielens_df.columns)
    test_df = pd.DataFrame(test_df, columns=movielens_df.columns)

    le_user = preprocessing.LabelEncoder()
    le_item = preprocessing.LabelEncoder()
    train_df["user_id_idx"] = le_user.fit_transform(train_df["user_id"].values)
    train_df["item_id_idx"] = le_item.fit_transform(train_df["item_id"].values)

    train_user_ids = train_df["user_id"].unique()
    train_item_ids = train_df["item_id"].unique()

    test_df = test_df[
        (test_df["user_id"].isin(train_user_ids)) & (test_df["item_id"].isin(train_item_ids))
    ]

    test_df["user_id_idx"] = le_user.transform(test_df["user_id"].values)
    test_df["item_id_idx"] = le_item.transform(test_df["item_id"].values)
    return train_df, test_df
