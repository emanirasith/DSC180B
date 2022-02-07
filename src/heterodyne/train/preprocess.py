from typing import Any, Callable, Generic, TypeVar

from pyspark.ml import Pipeline, Transformer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.param import Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit

from ..util import camel_case
from .transform import AbsoluteClipTransformer, FillNATransformer
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


SCALER_ARGS = {'withMean': True, 'withStd': True}


def build_scaler(
    input_col: str,
    output_col: str,
    with_mean=True,
    with_std=True,
):
    opts = dict(with_mean=with_mean, with_std=with_std)
    return StandardScaler(**{camel_case(k): v for k, v in opts.items()})


def build_statistics_pipeline(absolute_limit=10000) -> Pipeline:
    stages = [
        FillNATransformer(input_cols=list(STAT_COLS), scalar=0),
        AbsoluteClipTransformer(max_value=absolute_limit),
    ]
    stages.extend(build_scaler(col, col) for col in STAT_COLS)
    p = Pipeline(stages=stages)
    return p
