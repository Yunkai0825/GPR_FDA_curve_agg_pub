# operator_fusion_noweight/operator_fusion_plot_helpers.py
"""
Plot helpers for Operator Fusion aggregation.

Handles:
- Operator weight convergence plot

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Optional, Callable, TYPE_CHECKING

from ..summary_gpr_plotting import plot_weight_convergence

if TYPE_CHECKING:
    from ..pip3_summary_gpr_orchestrator import SummaryGPRResult


def plot_operator_weight_convergence(
    result: "SummaryGPRResult",
    output_dir: Path,
    verbose: bool = True,
) -> Optional[Path]:
    """Plot operator weight convergence if weight history is available."""
    if not result.operator_weight_history or len(result.operator_weight_history) <= 1:
        return None

    safe_key = result.group_key.replace('|', '_').replace('=', '_')
    iterations = np.arange(1, len(result.operator_weight_history) + 1)
    weight_history = np.vstack(result.operator_weight_history)
    op_weight_path = output_dir / f'Operator_Weight_Convergence_{safe_key}.png'

    return plot_weight_convergence(
        iterations=iterations,
        weight_history=weight_history,
        output_path=op_weight_path,
        group_key=result.group_key + " (operator)",
        verbose=verbose,
    )
