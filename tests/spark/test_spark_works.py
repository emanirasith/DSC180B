from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit

IRIS_COLUMNS = [
    'sepal_length',
    'sepal_width',
    'petal_length',
    'petal_width',
    'species',
]


def test_spark_works(iris_spark_df: DataFrame):
    assert set(iris_spark_df.columns) == set(IRIS_COLUMNS)
    assert iris_spark_df.columns == IRIS_COLUMNS
