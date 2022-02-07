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


# Ref: https://csyhuang.github.io/2020/08/01/custom-transformer/
class FillNATransformer(
    Transformer,
    HasInputCols,
    # HasOutputCols,
    DefaultParamsReadable,
    DefaultParamsWritable,
    Generic[T_Scalar],
):
    scalar = Param(placeholder(), 'scalar', 'The scalar fill value.')

    def __init__(
        self,
        *,
        input_cols: list[str] = None,
        # output_cols: list[str] = None,
        scalar: T_Scalar = None,
    ) -> None:
        super().__init__()
        # if not input_cols:
        #     raise ValueError(
        #         f'Expected input_cols, got falsy value {input_cols}'
        #     )
        self._setDefault(  # type: ignore
            input_cols=[],
            # output_cols=[],
            scalar=None,
        )

    # def set_params(
    #     self,
    #     *,
    #     input_cols: list[str] = None,
    #     # output_cols: list[str] = None,
    #     scalar: T_Scalar = None,
    # ):
    #     self._set(  # type: ignore
    #         input_cols=input_cols,
    #         # output_cols=output_cols,
    #         scalar=scalar,
    #     )

    def get_scalar(self) -> T_Scalar:
        return self.getOrDefault(self.scalar)

    def _transform(self, dataset: DataFrame) -> DataFrame:
        return dataset.fillna(self.get_scalar(), self.get_input_cols())


class ClipTransformer(
    Transformer,
    HasMinValue,
    HasMaxValue,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    def __init__(
        self,
        *,
        min_value: float = None,
        max_value: float = None,
    ) -> None:
        super().__init__()
        self._setDefault(min_value=min_value, max_value=max_value)  # type: ignore

    def _transform(self, dataset: DataFrame) -> DataFrame:
        min_value = lit(self.get_min_value())
        max_value = lit(self.get_max_value())
        cols = [
            case_when(
                col(name) < min_value,
                min_value,
                col(name) > max_value,
                max_value,
                col(name),
            ).alias(name)
            for name in dataset.columns
        ]
        return dataset.select(cols)


class AbsoluteClipTransformer(
    Transformer,
    HasMinValue,
    HasMaxValue,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    def __init__(
        self,
        *,
        max_value: float = None,
    ) -> None:
        super().__init__()
        self._setDefault(max_value=max_value)  # type: ignore

    def _transform(self, dataset: DataFrame) -> DataFrame:
        max_value = lit(self.get_max_value())
        cols = [
            case_when(
                F.abs(name) > max_value,
                max_value,
                col(name),
            ).alias(name)
            for name in dataset.columns
        ]
        return dataset.select(cols)
