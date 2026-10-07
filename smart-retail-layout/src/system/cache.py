"""Bộ nhớ đệm theo băm nội dung (kế hoạch v4 mục 1.5): FlowBank theo hash ShelfLayout + tham số luồng,
để vòng ngoài (tầng 1) và `rerun` không tính lại lưu lượng. Bộ nhớ trong + (tùy chọn) đĩa bằng pickle."""
from __future__ import annotations

import pickle
from pathlib import Path

from .contracts import _digest


def digest(*parts) -> str:
    return _digest(list(parts))


class Cache:
    def __init__(self, directory: str | Path | None = None, max_items: int = 64):
        self._mem: dict = {}
        self.dir = Path(directory) if directory else None
        self.max_items = max_items
        self.hits = 0
        self.misses = 0

    def get_or_compute(self, key: str, fn):
        if key in self._mem:
            self.hits += 1
            return self._mem[key]
        path = self.dir / f"{key}.pkl" if self.dir else None
        if path is not None and path.exists():
            self.hits += 1
            val = pickle.loads(path.read_bytes())
        else:
            self.misses += 1
            val = fn()
            if path is not None:
                self.dir.mkdir(parents=True, exist_ok=True)
                path.write_bytes(pickle.dumps(val))
        if len(self._mem) >= self.max_items:
            self._mem.pop(next(iter(self._mem)))
        self._mem[key] = val
        return val
