import pandas as pd
import pytest


@pytest.fixture
def iris_df(cache: pytest.Cache) -> pd.DataFrame:
    path = cache.makedir('data') / 'iris.feather'

    if path.exists():
        return pd.read_feather(path)

    iris: pd.DataFrame = pd.read_csv(  # type: ignore
        'https://github.com/scikit-learn/scikit-learn/raw/main/sklearn/datasets/data/iris.csv'
    )
    iris.to_feather(path)
    return iris
