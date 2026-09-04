from typing import Any

import numpy as np
from numpy.typing import NDArray


def as_2d_array(array: NDArray[Any], *, name: str) -> NDArray[Any]:
    """Return an array after confirming that it is two-dimensional."""
    result = np.asarray(array)
    if result.ndim != 2:
        raise ValueError(f"{name} must be a 2D array; got shape {result.shape}.")
    return result


def validate_same_shape(array: NDArray[Any],
                        expected: NDArray[Any],
                        *,
                        name: str,
                        expected_name: str,
                        ) -> None:
    """Require two arrays to have identical shapes."""
    if array.shape != expected.shape:
        raise ValueError(
            f"{name} must have the same shape as {expected_name}; got "
            f"{array.shape} and {expected.shape}.")


def validate_binary_array(array: NDArray[Any], *, description: str) -> None:
    """Require an array to contain only the binary values zero and one."""
    if not np.all((array == 0) | (array == 1)):
        raise ValueError(
            f"{description} must contain only binary values 0 and 1.")


def validate_positive_integer(value: int, *, name: str) -> None:
    """Require a non-boolean integer greater than or equal to one."""
    if (isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value < 1):
        raise ValueError(
            f"{name} must be an integer greater than or equal to 1.")
