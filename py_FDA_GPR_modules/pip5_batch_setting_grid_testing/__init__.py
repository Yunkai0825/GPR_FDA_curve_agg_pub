# pip5_batch_setting_grid_testing/__init__.py
"""
=============================================================================
Pipeline 5: Batch Settings Grid Testing
=============================================================================

This module provides batch testing infrastructure to run Summary GPR and
efficiency evaluation across all combinations of algorithm settings.

The default testing grid includes 6 non-redundant iterative combinations:
- normalize_summary: True/False
- weight_mode: 'equal'/'iterative'
- weight_scope: 'curve'/'point'

(Excludes equal+point as it's equivalent to equal+curve)

With ``include_fgpr=True``, 2 additional FGPR combinations are added
(normalized / un-normalized), giving 8 total.

Module Structure:
- batch_config.py: Configuration dataclasses
- batch_core.py: Core utilities (grid building, artifact copying)
- pip5_batch_grid_settings_orchestrator.py: High-level orchestrator
- batch_comparison_plotting.py: Overlay plots for comparing permutations

Author: Yunkai Sun (C-STEEL, CSE, ANL)
=============================================================================
"""

from .batch_config import (
    BatchDirParams,
    BatchTestingOptions,
    BatchRunConfig,
    ARTIFACT_PATTERNS,
)

from .batch_core import (
    build_testing_grid,
    discover_groupkeys,
    copy_artifacts,
    apply_options_to_config,
)

from .pip5_batch_grid_settings_orchestrator import (
    BatchTestingOrchestrator,
)

from .batch_comparison_plotting import (
    BatchComparisonConfig,
    PERMUTATION_COLORS,
    LINESTYLES,
    discover_permutations,
    load_summary_gpr_data,
    load_efficiency_data,
    load_efficiency_summary_csv,
    aggregate_efficiency_to_summary,
    parse_permutation_tag,
    get_short_label,
    export_summary_gpr_aggregate_csv,
    export_efficiency_aggregate_csv,
    export_all_aggregate_csvs,
    plot_summary_gpr_comparison,
    plot_efficiency_comparison,
    plot_combined_comparison,
    plot_all_comparisons,
)

__all__ = [
    # Config
    "BatchDirParams",
    "BatchTestingOptions",
    "BatchRunConfig",
    "ARTIFACT_PATTERNS",
    # Core utilities
    "build_testing_grid",
    "discover_groupkeys",
    "copy_artifacts",
    "apply_options_to_config",
    # Orchestrator
    "BatchTestingOrchestrator",
    # Comparison plotting
    "BatchComparisonConfig",
    "PERMUTATION_COLORS",
    "LINESTYLES",
    "discover_permutations",
    "load_summary_gpr_data",
    "load_efficiency_data",
    "load_efficiency_summary_csv",
    "aggregate_efficiency_to_summary",
    "parse_permutation_tag",
    "get_short_label",
    # CSV export functions
    "export_summary_gpr_aggregate_csv",
    "export_efficiency_aggregate_csv",
    "export_all_aggregate_csvs",
    # Plot functions
    "plot_summary_gpr_comparison",
    "plot_efficiency_comparison",
    "plot_combined_comparison",
    "plot_all_comparisons",
]
