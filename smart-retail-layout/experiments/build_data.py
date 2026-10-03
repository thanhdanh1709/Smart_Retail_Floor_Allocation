"""Bước dữ liệu: ước lượng tham số từ Instacart (B3) và lưu 9 + 1 mặt bằng (B7.1)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import params  # noqa: E402
from src.instance import save_floorplans  # noqa: E402


def main():
    print("1) Tham số từ dữ liệu giỏ hàng Instacart ...")
    params.build_all()
    print("2) Lưu mặt bằng chuẩn ...")
    for p in save_floorplans():
        print("  ", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
