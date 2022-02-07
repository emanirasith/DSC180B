from typing import Any, Callable, Generic, TypeVar

from pyspark.ml import Transformer
from pyspark.ml.param import Param, Params, TypeConverters

# from pyspark.ml.param.shared import HasMax
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit

from .util import (
    HasInputCols,
    HasMaxValue,
    HasMinValue,
    HasOutputCols,
    HasScalarValue,
    T,
    T_Numeric,
    T_Scalar,
    case_when,
    placeholder,
)

STAT_COLS = (
    'total_vals',
    'num_nans',
    '%_nans',
    'num_of_dist_val',
    '%_dist_val',
    'mean',
    'std_dev',
    'min_val',
    'max_val',
    'has_delimiters',
    'has_url',
    'has_email',
    'has_date',
    'mean_word_count',
    'std_dev_word_count',
    'mean_stopword_total',
    'stdev_stopword_total',
    'mean_char_count',
    'stdev_char_count',
    'mean_whitespace_count',
    'stdev_whitespace_count',
    'mean_delim_count',
    'stdev_delim_count',
    'is_list',
    'is_long_sentence',
)


def _abs_clip(limit: float) -> Callable[[Column], Column]:
    def inner(col: Column):
        return F.when(F.abs(col) > limit, F.signum(col) * limit).otherwise(col)

    return inner


def _process_stats_old(
    df: DataFrame,
    normalize=False,
    abs_limit: float = 10000,
) -> DataFrame:
    df = df.select(*STAT_COLS).fillna(0)

    if normalize:
        clip = _abs_clip(abs_limit)
        df = df.select([clip(col(name)) for name in df.columns])
        # Standard Scaler omitted

    return df


def _extract_features_old(
    df: DataFrame,
):
    pass
