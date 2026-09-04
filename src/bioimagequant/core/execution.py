from collections.abc import Callable, Iterable
from typing import Any, TypeVar, cast

from joblib import Parallel, delayed

from bioimagequant.core.validation import validate_positive_integer


MEMMAP_THRESHOLD = "10M"
Result = TypeVar("Result")
Job = tuple[Any, ...]


def run_jobs(function: Callable[..., Result],
             jobs: Iterable[Job],
             *,
             workers: int,
             ) -> list[Result]:
    """Run argument tuples serially or in memory-mapped worker processes."""
    validate_positive_integer(workers, name="workers")
    if workers == 1:
        return [function(*job) for job in jobs]
    return cast(list[Result], Parallel(
        n_jobs=workers,
        backend="loky",
        max_nbytes=MEMMAP_THRESHOLD,
        mmap_mode="r",
    )(
        delayed(function)(*job)
        for job in jobs))
