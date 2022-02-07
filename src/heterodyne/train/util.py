from typing import Generic, ParamSpec, TypeVar

from pyspark.ml.param.shared import Param, Params, TypeConverters
from pyspark.sql import Column
from pyspark.sql import functions as F

P = ParamSpec('P')
R = TypeVar('R')
T = TypeVar('T')

Scalar = bool | int | float | str
T_Scalar = TypeVar('T_Scalar', bool, int, float, str)
T_Numeric = TypeVar('T_Numeric', int, float)


def placeholder() -> Params:
    return Params._dummy()  # type: ignore


class HasInputCol(Params):
    """
    Mixin for param input_col: input column name.
    """

    input_col = Param(
        placeholder(),
        'input_col',
        'input column name',
        typeConverter=TypeConverters.toString,
    )

    def __init__(self):
        super().__init__()

    def get_input_col(self):
        """
        Gets the value of input_col or its default value.
        """
        return self.getOrDefault(self.input_col)


class HasInputCols(Params):
    """
    Mixin for param input_cols: input column names.
    """

    input_cols = Param(
        placeholder(),
        'input_cols',
        'input column names',
        typeConverter=TypeConverters.toListString,
    )

    def __init__(self):
        super().__init__()

    def get_input_cols(self):
        """
        Gets the value of input_cols or its default value.
        """
        return self.getOrDefault(self.input_cols)


class HasOutputCol(Params):
    """
    Mixin for param output_col: output column name.
    """

    output_col = Param(
        placeholder(),
        'output_col',
        'output column name',
        typeConverter=TypeConverters.toString,
    )

    def __init__(self):
        super().__init__()
        self._setDefault(output_col=self.uid + '__output')  # type: ignore

    def get_output_col(self):
        """
        Gets the value of output_col or its default value.
        """
        return self.getOrDefault(self.output_col)


class HasOutputCols(Params):
    """
    Mixin for param output_cols: output column names.
    """

    output_cols = Param(
        placeholder(),
        'output_cols',
        'output column names',
        typeConverter=TypeConverters.toListString,
    )

    def __init__(self):
        super().__init__()

    def get_output_cols(self):
        """
        Gets the value of output_cols or its default value.
        """
        return self.getOrDefault(self.output_cols)


class HasScalarValue(Generic[T_Scalar], Params):
    """
    Mixin for param scalar_value: scalar value.
    """

    scalar_value = Param(
        placeholder(),
        'scalar_value',
        'scalar value',
    )

    def __init__(self):
        super().__init__()

    def get_scalar_value(self) -> T_Scalar:
        """
        Gets the value of scalar_value or its default value.
        """
        return self.getOrDefault(self.scalar_value)


class HasMinValue(Params):
    """
    Mixin for param min_value: minimum value.
    """

    min_value = Param(
        placeholder(),
        'min_value',
        'minimum value',
        typeConverter=TypeConverters.toFloat,
    )

    def __init__(self):
        super().__init__()

    def get_min_value(self) -> float:
        """
        Gets the value of min_value or its default value.
        """
        return self.getOrDefault(self.min_value)


class HasMaxValue(Params):
    """
    Mixin for param max_value: maximum value.
    """

    max_value = Param(
        placeholder(),
        'max_value',
        'maximum value',
        typeConverter=TypeConverters.toFloat,
    )

    def __init__(self):
        super().__init__()

    def get_max_value(self) -> float:
        """
        Gets the value of max_value or its default value.
        """
        return self.getOrDefault(self.max_value)


def case_when(*args: Column):
    l = len(args)
    *case_args, otherwise = args

    if not l % 2:
        raise ValueError(f'Expected an odd number of arguments, got {l}')
    if not l >= 3:
        raise ValueError(f'Expected at least three arguments, got {l}')

    expr = F.when(case_args[0], case_args[1])

    if l == 2:
        return expr.otherwise(otherwise)

    when = case_args[2::2]
    values = case_args[3::2]

    for left, right in zip(when, values):
        expr = expr.when(left, right)

    return expr.otherwise(otherwise)
