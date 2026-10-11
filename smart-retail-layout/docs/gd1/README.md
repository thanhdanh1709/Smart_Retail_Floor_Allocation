# GĐ1 — System contracts and v3 bridge

GĐ1 adds a versioned `src/system/` layer around the existing v3 floorplan, GA, payoff, and customer simulator. The bridge is an integration prototype for L0-shaped inputs. It does not implement the future analytic L0 flow, online picker routing, SKU-level SSAP, or real-store calibration.

## Run a quick design

Run commands from `smart-retail-layout/` in the project Python environment:

```powershell
python -m src.system design --spec specs/store.yaml --quick
python -m src.system design --spec specs/case_minimart.yaml --quick
```

The default result is written to `results/GD1/runs/<case_id>/<content_hash>/design.json`. The content cache is isolated in `results/GD1/cache/`. Use `--no-cache` to recompute, or `--out <directory-or-json-file>` for a separate output. An existing output with another content hash is never overwritten.

Validation failures are strict JSON on stderr and exit with code 2. Stage failures are strict JSON and exit with code 3. Successful designs exit with code 0. The CLI applies a deterministic quick profile: payoff GA at most 16 individuals × 15 generations, final GA at most 20 × 25 generations, no wall-clock stopping or local search, and at most 100 simulated customers.

## Contract identifiers and units

- `schema_version` is `1.0`; GĐ1 accepts only `fidelity: L0` and `execution_mode: v3_bridge`.
- Category IDs come from the selected processed `categories.csv` records, such as `G01`. Array indexes remain internal.
- Slot IDs are `K001...` within a layout. Their public identity is the pair `(layout_id, slot_id)`. The result preserves both floorplan slot index and instance slot index for traceability.
- `ShelfLayout.d0k_m` follows full `FloorPlan.slots` order, and remains null with `pickup_cell` when no staging point is explicitly configured. Instance-indexed consumers must use `inst.fp.require_pickup()[inst.slot_idx]`.
- Synthetic SKUs use IDs such as `SYN-G01-001`; they cannot be confused with observed products. Their `frequency_weight` is seeded fixture data for integration testing, not measured demand.
- Floorplan cells are 1 metre square in the existing v3 implementation. A different `cell_size_m` is rejected because v3 geometry assumes one metre per cell.
- `distance_m`, `trip_time_s`, exposed slots/customer, items/customer, and congestion cells use explicit units in the metric registry. `trip_time_min` from v3 is converted to seconds/customer.
- QAP `z1` is a model cost proxy, not metres. `z2`, `impulse_value_proxy`, and `basket_value_proxy` are arbitrary assumed-value units, never VND or observed revenue. Every metric records its direction and source.
- Missing/non-finite optional metric values serialize as JSON `null` with `null_reason`; JSON export rejects NaN and infinities.

## Four temporary layers

1. **T1:** resolve a preset through `layouts.PRESETS` or load a supplied grid file, then build the existing `Instance`. The minimart case uses the actual `case_minimart.txt` grid. No pickup cell is inferred from `E` or `C`; `pickup_cell` stays null unless explicitly supplied.
2. **T2:** compute the v3 heuristic payoff using the configured small GA budget, run `ga.run`, then check the full v3 feasibility result (compatibility, separation, relocation budget). If GA is infeasible, the existing current layout is an eligible fallback only if it passes the same feasibility check. An infeasible result skips T3/T4.
3. **T3:** generate a deterministic synthetic frequency fixture and rank its virtual products within the category slot selected by T2. No physical width, facings, capacity, price, or sales are fabricated. The item-to-category feedback loop is explicitly `not_implemented`.
4. **T4:** run the existing `Simulator` on processed category baskets and the selected category assignment. It does not consume synthetic SKU records or model picker flows. `flow_report.model` is `v3_customer_simulation` and its metrics remain modeled proxies.

Every emitted layer declares a method, status, and data class. The result validator checks IDs across demand → category assignment → slot → synthetic product → virtual placement, verifies every accepted assignment has zero violations, and checks metric source/unit/direction/data class against the registry.

`case_minimart.yaml` sets `data.level: group` explicitly. The `make_instance()` API otherwise tries to infer a level from `scale`, but `minimart` is a file-layout label and not a supported data scale. `processed_root` points to `data/processed`; `params.load()` appends the configured level directory.

## Reproducibility and cache

The system keeps separate input, cache, and result identities:

- `input_hash` hashes the resolved effective config, effective grid, and source data SHA-256 values, including `categories.csv`, `w.npy`, `f.npy`, `lift.npy`, `p.npy`, `v.npy`, and `baskets.npz`.
- `cache_key` also includes the relevant source-file fingerprint, schema/stage version, and Python/platform/dependency fingerprint. A change to layout, processed data, seed/config, code, or environment invalidates the cache.
- `content_hash` hashes deterministic result fields and the input/data/source identities. It excludes runtime, timestamp, cache-hit state, Git SHA, output path, and environment metadata. Cold and warm runs therefore have the same content hash.

The Git SHA is retained for traceability but is not used alone: this checkout contains valid uncommitted GĐ0 changes. Cache reads reject non-standard JSON constants and revalidate the entire StoreDesign contract before returning a hit. Cache writes are atomic and isolated under `results/GD1/cache/`. An existing export is accepted only if its stored result passes validation and its content hash matches its actual content.

## Verification

```powershell
python -m pytest -q -p no:cacheprovider tests/test_system_contracts.py
python -m pytest -q -p no:cacheprovider -m e2e tests/test_system_e2e.py
python -m pytest -q -p no:cacheprovider -m e2e
python -m pytest -q -p no:cacheprovider
python -m pip check
```

The E2E suite runs both `grid_small_l0` and `case_minimart_l0`, checks IDs, units, feasibility, strict JSON, bounded runtime, and content-hash stability on a cache hit. These are software integration checks on synthetic/proxy data; they are not evidence for GĐ7 or real store performance.
