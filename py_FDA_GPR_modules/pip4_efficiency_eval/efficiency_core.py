# pip4_efficiency_eval/efficiency_core.py
"""
Core algorithms for Efficiency Evaluation.

This module re-exports from the separated submodules for backward compatibility:
- mc_sampling: Monte Carlo sampling utilities
- learning_curve: Learning curve computation

And provides high-level processing functions:
- process_potential_learning_curve: Process learning curve for a potential group

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

from typing import List, Optional, TYPE_CHECKING

# Re-export from mc_sampling
from .mc_sampling import (
    bounded_comb,
    mc_repeats,
    balanced_subset,
    generate_balanced_subsets,
)

# Re-export from learning_curve
from .learning_curve import (
    SubsetResult,
    LearningCurveResult,
    fast_summary_gpr_core,
    error_metric,
    learning_curve,
    learning_curve_layered,
    summarize_learning_curve,
)

if TYPE_CHECKING:
    from ..pip3_FDA_scoring_and_aggregations import IndividualGPRData
    from .efficiency_config import GlobalParams, ScaleParams

# Keep backward compatibility alias
_balanced_subset = balanced_subset


def process_potential_learning_curve(
    gpr_list: List["IndividualGPRData"],
    summary_gpr_config,
    summary_gpr_hyperparams,
    globpara: Optional["GlobalParams"] = None,
    scapara: Optional["ScaleParams"] = None,
    verbose: bool = True,
    aggregation_method: str = "iterative",
    layered: bool = False,
    output_dir: Optional[str] = None,
    csv_stem: Optional[str] = None,
    plot_callback=None,
) -> Optional[LearningCurveResult]:
    """
    Process learning curve for a single potential group.
    
    This is the main computation function that:
    1. Computes full-data reference using all curves
    2. Defines subset sizes (every 2 curves from 2 to N)
    3. Runs learning curve analysis across subset sizes
    
    Parameters
    ----------
    gpr_list : List[IndividualGPRData]
        List of IndividualGPRData objects for this potential.
    summary_gpr_config : SummaryGPRConfig
        Configuration for summary GPR algorithm.
    summary_gpr_hyperparams : SummaryGPRHyperParams
        Hyperparameters for summary GPR.
    globpara : GlobalParams, optional
        Global experiment parameters.
    scapara : ScaleParams, optional
        Scaling parameters.
    verbose : bool
        Print progress.
    aggregation_method : str
        "iterative" for variance-based methods, "fgpr" for functional GPR.
    layered : bool
        If True, use layer-by-layer MC iteration with real-time CSV
        persistence and resume support.
    output_dir : str, optional
        Output directory (required when ``layered=True``).
    csv_stem : str, optional
        Base name for CSV files (required when ``layered=True``).
    plot_callback : callable, optional
        ``plot_callback(df_detailed, layer)`` called at logarithmically-
        spaced layer numbers to generate an interim learning-curve figure.
        
    Returns
    -------
    LearningCurveResult or None
        Learning curve results (summary + detailed DataFrames), 
        or None if not enough curves (< 3).
    """
    from .efficiency_config import GlobalParams, ScaleParams
    
    globpara = globpara or GlobalParams()
    scapara = scapara or ScaleParams()

    # ----- Pre-filter outlier curves with near-zero scale factors -----
    # Must happen BEFORE computing N / subset sizes / MC draws so that
    # every subset of size m actually contains m usable curves.
    ratio_thresh = getattr(
        summary_gpr_hyperparams, 'fgpr_min_scale_factor_ratio', 0.0
    )
    if ratio_thresh > 0 and len(gpr_list) > 1:
        import numpy as _np
        abs_factors = _np.array([
            abs(gpr.y_scaling.params.get("factor", 1.0))
            for gpr in gpr_list
        ])
        median_factor = float(_np.median(abs_factors))
        min_factor = ratio_thresh * median_factor
        keep_mask = abs_factors >= min_factor
        n_excluded = int((~keep_mask).sum())
        if n_excluded > 0:
            keep_idx = _np.where(keep_mask)[0].tolist()
            if verbose:
                excluded_ids = [
                    gpr_list[i].sample_id
                    for i in range(len(gpr_list))
                    if not keep_mask[i]
                ]
                print(
                    f"  Pre-filtered {n_excluded} outlier curve(s) "
                    f"(|factor| < {min_factor:.4g}, "
                    f"median={median_factor:.4g}): {excluded_ids}"
                )
            gpr_list = [gpr_list[i] for i in keep_idx]

    N = len(gpr_list)
    if N < 3:
        if verbose:
            print(f"  Only {N} curves after filtering, skipping (need >= 3)")
        return None
    
    if verbose:
        print(f"  Processing {N} curves (after outlier filtering)...")
    
    # Compute reference (full data)
    if verbose:
        print("    Computing full-data reference...")
    ref_result = fast_summary_gpr_core(
        gpr_list,
        summary_gpr_config=summary_gpr_config,
        summary_gpr_hyperparams=summary_gpr_hyperparams,
        aggregation_method=aggregation_method,
    )
    ref_mean = ref_result.y_mean
    ref_bar = ref_result.y_bar
    
    if verbose:
        print(f"    Reference: {N} curves, {ref_result.n_iterations} iterations")
    
    # Define subset sizes (every 2 curves from 2 to N)
    sizes = list(range(2, N, 2)) + [N]
    
    # Compute total MC iterations for progress estimation
    if verbose:
        total_mc = 0
        for ss in sizes:
            if ss == N:
                total_mc += 1
            else:
                total_mc += mc_repeats(N, ss, globpara=globpara)
        print(f"    Subset sizes: {len(sizes)} levels  "
              f"(min={sizes[0]}, max={sizes[-1]})")
        print(f"    Total MC combinations: {total_mc}  "
              f"(method={aggregation_method})")
    
    # Compute learning curve
    if verbose:
        mode_str = "layer-by-layer" if layered else "size-by-size"
        print(f"    Computing learning curve ({mode_str}) "
              f"for {len(sizes)} subset sizes...", flush=True)

    if layered:
        if output_dir is None or csv_stem is None:
            raise ValueError(
                "output_dir and csv_stem are required for layered mode"
            )
        lc_result = learning_curve_layered(
            gpr_list,
            ref_mean,
            ref_bar,
            sizes,
            summary_gpr_config=summary_gpr_config,
            summary_gpr_hyperparams=summary_gpr_hyperparams,
            globpara=globpara,
            scapara=scapara,
            output_dir=output_dir,
            csv_stem=csv_stem,
            verbose=verbose,
            aggregation_method=aggregation_method,
            resume=True,
            plot_callback=plot_callback,
        )
    else:
        lc_result = learning_curve(
            gpr_list,
            ref_mean,
            ref_bar,
            sizes,
            summary_gpr_config=summary_gpr_config,
            summary_gpr_hyperparams=summary_gpr_hyperparams,
            globpara=globpara,
            scapara=scapara,
            verbose=verbose,
            aggregation_method=aggregation_method,
        )

    return lc_result


__all__ = [
    # MC Sampling
    "bounded_comb",
    "mc_repeats",
    "balanced_subset",
    "generate_balanced_subsets",
    # Learning Curve
    "SubsetResult",
    "LearningCurveResult",
    "fast_summary_gpr_core",
    "error_metric",
    "learning_curve",
    "learning_curve_layered",
    "summarize_learning_curve",
    # High-level processing
    "process_potential_learning_curve",
]
