"""Bước dữ liệu: ước lượng tham số từ Instacart (B3) và lưu 9 + 1 mặt bằng (B7.1)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import params  # noqa: E402
from src.instance import save_floorplans  # noqa: E402


def main():
    if "--items" in sys.argv:            # v4: chỉ dựng dữ liệu cấp món + hồ sơ giờ
        from src import items, schedule
        print("Dữ liệu cấp món ...")
        items.build()
        print("Hồ sơ theo giờ ...")
        print(schedule.build_hour_profile().round(4).to_string(index=False))
        return
    print("1) Tham số từ dữ liệu giỏ hàng Instacart ...")
    params.build_all()
    print("2) Lưu mặt bằng chuẩn ...")
    for p in save_floorplans():
        print("  ", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
