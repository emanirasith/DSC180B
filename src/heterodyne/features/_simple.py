from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit

from .util import ColumnFn, is_struct_field_numeric

SIMPLE_FEATURES: dict[str, ColumnFn] = {
    'count': F.count,
    'distinct': F.count_distinct,
    'distinct_percent': lambda c: 100 * F.count_distinct(c) / F.count(c),
}
SIMPLE_NUMERIC_FEATURES: dict[str, ColumnFn] = {
    'nans': lambda c: F.count(F.isnan(c)),
    'nans_percent': lambda c: 100 * F.count(F.isnan(c)) / F.count(c),
    'mean': F.mean,
    'std': F.stddev,
    'min': F.min,
    'max': F.max,
}

LEGACY_NAME_MAPPING: dict[str, str] = {
    'count': 'total_vals',
    'distinct': 'num_of_dist_val',
    'distinct_percent': '%_dist_val',
    'nans': 'num_nans',
    'nans_percent': '%_nans',
    'mean': 'mean',
    'std': 'std_dev',
    'min': 'min_val',
    'max': 'max_val',
}

N_SAMPLES = 5


def simple_features(df: SparkDataFrame) -> SparkDataFrame:
    simple_aggs = [
        fn(col(c)).alias(f'{c}::{name}')
        for c in df.columns
        for name, fn in SIMPLE_FEATURES.items()
    ]

    numeric_aggs = [
        (fn(col(c)) if is_struct_field_numeric(df.schema[c]) else lit(0)).alias(
            f'{c}::{name}'
        )
        for c in df.columns
        for name, fn in SIMPLE_NUMERIC_FEATURES.items()
    ]

    agg_df = df.agg(*simple_aggs, *numeric_aggs)
    return agg_df
