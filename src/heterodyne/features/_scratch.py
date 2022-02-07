from itertools import chain
from typing import TYPE_CHECKING, Callable, Iterable, cast

from pyspark import Row
from pyspark.sql import Column
from pyspark.sql import DataFrame as SparkDataFrame
from pyspark.sql import functions as F
from pyspark.sql.functions import col, lit
from pyspark.sql.types import StringType, StructField, StructType

from .util import ColumnFn, is_struct_field_numeric

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    from pyspark.sql.pandas._typing import DataFrameLike

'Record_id,Attribute_name,y_act,total_vals,num_nans,%_nans,num_of_dist_val,%_dist_val,mean,std_dev,min_val,max_val,sample_1,sample_2,sample_3,sample_4,sample_5,has_delimiters,has_url,has_email,has_date,mean_word_count,std_dev_word_count,mean_stopword_total,stdev_stopword_total,mean_char_count,stdev_char_count,mean_whitespace_count,stdev_whitespace_count,mean_delim_count,stdev_delim_count,is_list,is_long_sentence'


SIMPLE_FEATURES: dict[str, ColumnFn] = {
    'total_vals': F.count,
    'num_of_dist_val': F.count_distinct,
    '%_dist_val': lambda c: 100 * F.count_distinct(c) / F.count(c),
}
SIMPLE_NUMERIC_FEATURES: dict[str, ColumnFn] = {
    'num_nans': lambda c: F.count(F.isnan(c)),
    '%_nans': lambda c: 100 * F.count(F.isnan(c)) / F.count(c),
    'mean': F.mean,
    'std_dev': F.stddev,
    'min_val': F.min,
    'max_val': F.max,
}


def simple_summary(df: SparkDataFrame) -> SparkDataFrame:
    cols = [col(name) for name in df.columns]
    return df.agg(*(F.count_distinct(c) for c in cols))


def _aggs(
    fns: dict[str, ColumnFn],
    columns: list[str],
) -> list[Column]:
    return [
        fn(col(c)).alias(f'{c}::{name}')
        for c in columns
        for name, fn in fns.items()
    ]


def sample(df: SparkDataFrame):
    count = df.count()
    aggs = [F.collect_list(col(name)) for name in df.columns]
    return df.sample(fraction=5 / count * 2).agg(*aggs)


# TODO: Re-implement in Java
def _sample_impl_python_map_partitions(
    _: int,
    rows: Iterable[Row],
) -> Iterable[Row]:
    rows = iter(rows)
    row = next(rows)

    fields: list[str] = row.__fields__
    samples: dict[str, list] = {name: [] for name in fields}
    done = [False] * len(fields)

    for row in chain([row], rows):
        for i, (k, v) in enumerate(zip(fields, row)):
            if done[i]:
                continue
            l = samples[k]
            if len(l) >= 5:
                continue
            if v and v not in l:
                l.append(v)
                if len(l) >= 5:
                    done[i] = True

        if all(done):
            break
    results = [
        Row(field, v)  # type: ignore
        for field, values in samples.items()
        for v in values
    ]
    return results


# def create_accumulator(sc: SparkContext):

# TODO: Optimize using basic statistics for high/low cardinality columns
def _sample_with_rdd(
    df: SparkDataFrame,
    subset: list[str] = None,
    # sc: SparkContext = None,
):
    df.mapInPandas
    col_set = set(df.columns)
    # if subset:
    #     df = df.select([name for name in subset if name in col_set])
    # NOTE: Assumes that 5 * n_partitions rows fits in memory
    rdd = df.rdd
    rdd = rdd.mapPartitionsWithIndex(_sample_impl_python_map_partitions)
    df = rdd.toDF(['field', 'value'])
    # if subset:
    #     if not sc:
    #         sc = SparkContext.getOrCreate()
    #     sc.parallelize(
    #         [
    #             Row()
    #         ]
    #     )
    rows = rdd.collect()

    fields = df.columns

    # .reduceByKey()
    # df.rdd.flatMapValues
    # df.rdd.reduce()


_UNIQUE_LIMIT = 5


def _sample_impl_map_in_pandas(
    n: int,
) -> Callable[[Iterable[pd.DataFrame]], Iterable[pd.DataFrame]]:
    def _impl(
        pdf_it: Iterable[pd.DataFrame],
    ) -> Iterable[pd.DataFrame]:
        samples: dict[str, set] | None = None
        for pdf in pdf_it:
            if not samples:
                samples = {name: set() for name in pdf.columns}
            if all(len(sample_set) >= 5 for sample_set in samples.values()):
                break
            for name in pdf.columns:
                s = samples[name]
                u: np.ndarray = pdf[name].unique()  # type: ignore
                s.update(u[: n - len(s)].tolist())

        if not samples:
            return

        yield pd.DataFrame.from_records(
            (
                (field, value)
                for field, values in samples.items()
                for value in values
            ),
            columns=['column', 'value'],
        )

    return _impl


# TODO: Optimize using basic statistics for high/low cardinality columns
def _sample_with_pandas(df: SparkDataFrame, n: int = 5) -> SparkDataFrame:
    fn = _sample_impl_map_in_pandas(n)
    fn = cast(
        Callable[[Iterable['DataFrameLike']], Iterable['DataFrameLike']],
        fn,
    )
    df.mapInPandas(
        fn,
        StructType(
            [
                StructField('column', StringType()),
                StructField('value', StringType()),
            ],
        ),
    )
    return df
