"""Shared tqdm helpers for synchronous and asynchronous workflow progress."""

import asyncio
from collections.abc import Awaitable, Iterable, Sequence
from typing import TypeVar, cast

from tqdm.auto import tqdm

T = TypeVar("T")


def make_progress_bar(
    iterable: Iterable[T] | None = None,
    *,
    total: int | None = None,
    desc: str,
    unit: str,
    leave: bool = False,
) -> tqdm:
    """Creates a consistently configured tqdm progress bar."""

    return tqdm(
        iterable=iterable,
        total=total,
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
        leave=leave,
    )


async def _await_with_index(index: int, awaitable: Awaitable[T]) -> tuple[int, T]:
    """Wraps an awaitable so progress updates can preserve result ordering."""

    return index, await awaitable


async def gather_with_progress(
    awaitables: Sequence[Awaitable[T]],
    *,
    desc: str,
    unit: str,
) -> list[T]:
    """Awaits tasks with a progress bar while preserving input order."""

    if not awaitables:
        return []

    tasks = [
        asyncio.create_task(_await_with_index(index, awaitable))
        for index, awaitable in enumerate(awaitables)
    ]
    results: list[T | None] = [None] * len(tasks)
    with make_progress_bar(total=len(tasks), desc=desc, unit=unit) as progress_bar:
        for task in asyncio.as_completed(tasks):
            index, value = await task
            results[index] = value
            progress_bar.update(1)
    return [cast(T, value) for value in results]
