"""Versioned contracts and orchestration for the v4 system layer.

GĐ1 deliberately wraps the v3 algorithms. It does not claim that the
temporary L0 bridge is a validated retail decision model.
"""

from .contracts import StoreDesign, StoreSpec
from .orchestrator import design_store
from .t2 import build_twoflow_inputs, run_t2

__all__ = ["StoreDesign", "StoreSpec", "design_store", "build_twoflow_inputs", "run_t2"]
