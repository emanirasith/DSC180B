from functools import wraps
from typing import Callable, Concatenate, ParamSpec, TypeVar

from pyspark.ml.param.shared import Params

P = ParamSpec('P')
R = TypeVar('R')
T = TypeVar('T')


def placeholder() -> Params:
    return Params._dummy()  # type: ignore


def kwargs_only(
    func: Callable[Concatenate[T, P], R]
) -> Callable[Concatenate[T, P], R]:
    """
    A decorator that forces keyword arguments in the wrapped method
    and saves actual input keyword arguments in `_input_kwargs`.

    Notes
    -----
    Should only be used to wrap a method where first arg is `self`
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if len(args) > 0:
            raise TypeError(
                'Method %s forces keyword arguments' % func.__name__
            )
        self._input_kwargs = kwargs
        return func(self, **kwargs)

    return wrapper
