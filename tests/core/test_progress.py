"""Tests for shared tqdm progress helpers."""

import asyncio

from src.core import progress


class DummyProgressBar:
    """Minimal progress bar double for helper tests."""

    def __init__(self) -> None:
        self.updates = 0

    def __enter__(self) -> "DummyProgressBar":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def update(self, amount: int = 1) -> None:
        self.updates += amount


async def _delayed_value(value: str, delay: float) -> str:
    """Returns a value after a small async delay."""

    await asyncio.sleep(delay)
    return value


def test_gather_with_progress_preserves_result_order(
    monkeypatch,
) -> None:
    """Completed tasks should be reordered back to the original input order."""

    bar = DummyProgressBar()
    monkeypatch.setattr(progress, "make_progress_bar", lambda **_kwargs: bar)

    results = asyncio.run(
        progress.gather_with_progress(
            [
                _delayed_value("first", 0.01),
                _delayed_value("second", 0.0),
                _delayed_value("third", 0.005),
            ],
            desc="Patent checks",
            unit="mol",
        )
    )

    assert results == ["first", "second", "third"]
    assert bar.updates == 3
