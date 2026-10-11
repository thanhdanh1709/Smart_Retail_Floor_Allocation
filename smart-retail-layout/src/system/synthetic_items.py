"""Seeded synthetic SKU records for exercising the item contract only."""
from __future__ import annotations

from .contracts import CategoryPlan, ItemFixture, ProductRecord


def generate_item_fixture(plan: CategoryPlan, *, case_id: str, seed: int,
                          items_per_category: int, generator_version: int) -> ItemFixture:
    if generator_version != 1:
        raise ValueError(f"unsupported synthetic item generator version: {generator_version}")
    import numpy as np

    rng = np.random.default_rng(seed)
    items: list[ProductRecord] = []
    for category_id in sorted(item.category_id for item in plan.assignments):
        weights = rng.uniform(0.01, 1.0, size=items_per_category)
        order = sorted(range(items_per_category), key=lambda index: (-weights[index], index))
        rank_of = {index: rank for rank, index in enumerate(order, start=1)}
        for index, weight in enumerate(weights, start=1):
            items.append(ProductRecord(
                sku_id=f"SYN-{category_id}-{index:03d}",
                category_id=category_id,
                frequency_weight=float(weight),
                rank=rank_of[index - 1],
            ))
    return ItemFixture(
        fixture_id=f"synthetic_frequency_v{generator_version}_{case_id}_{seed}",
        generator_version=generator_version,
        seed=seed,
        products=tuple(items),
        provenance=(
            "Generated synthetic frequency weights for contract integration only; "
            "not observed SKU demand, capacity, price, sales, or physical planogram data."
        ),
    )
