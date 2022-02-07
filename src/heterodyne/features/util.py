from dataclasses import dataclass
from typing import Callable, overload

from pyspark.sql import Column, DataFrame
from pyspark.sql.functions import col, lit
from pyspark.sql.types import NumericType, StructField, StructType

ColumnFn = Callable[[Column], Column]


@dataclass
class ColumnAgg:
    name: str
    func: ColumnFn

    def __call__(self, col: Column) -> Column:
        return self.func(col)


@overload
def column_agg(
    name: str,
) -> Callable[[ColumnFn], ColumnAgg]:
    ...


@overload
def column_agg(func: ColumnFn, /) -> ColumnAgg:
    ...


def column_agg(name: ColumnFn | str):
    if callable(name):
        result = column_agg(name.__name__)(name)
        return result
    name_: str = name

    def _inner(func: ColumnFn, /) -> ColumnAgg:
        return ColumnAgg(name_, func)

    return _inner


def col_map(func: ColumnFn, df: DataFrame) -> list[Column]:
    return [func(col(name)) for name in df.columns]


def is_struct_field_numeric(sf: StructField) -> bool:
    return isinstance(sf.dataType, NumericType)


def require_numeric(df: DataFrame, col_name: str, fn: ColumnFn) -> ColumnFn:
    if not is_struct_field_numeric(df.schema[col_name]):
        return lambda _: lit(0)
    return fn


def aggregate_columns(
    fns: dict[str, ColumnFn],
    columns: list[str],
) -> list[Column]:
    return [
        fn(col(c)).alias(f'{c}::{name}')
        for c in columns
        for name, fn in fns.items()
    ]
