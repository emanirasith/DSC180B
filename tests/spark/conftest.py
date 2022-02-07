from typing import Iterable

import pandas as pd
import pytest
from pyspark import SparkContext
from pyspark.sql import DataFrame, SparkSession

# @pytest.fixture(scope='session')
# def spark_context() -> Iterable[SparkContext]:
#     sc = SparkContext.getOrCreate()
#     yield sc
#     sc.stop()


@pytest.fixture(name='spark', scope='session')
def spark_session(
    # spark_context: SparkContext,
) -> Iterable[SparkSession]:
    yield SparkSession.builder.getOrCreate()


@pytest.fixture
def iris_spark_df(spark: SparkSession, iris_df: pd.DataFrame) -> DataFrame:
    return spark.createDataFrame(iris_df)
