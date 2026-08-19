# pip5_batch_setting_grid_testing/batch_core.py
"""
Core utilities for batch grid testing.

Provides functions for:
- Building the testing grid
- Discovering potentials from files
- Copying artifacts

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import os
import glob
import shutil
from pathlib import Path
from typing import List, Optional

from .batch_config import BatchTestingOptions, ARTIFACT_PATTERNS


def build_testing_grid(
    include_fgpr: bool = False,
    include_student_t: bool = False,
) -> List[BatchTestingOptions]:
    """
    Build the non-redundant flag combinations for GPR-FDA.
    
    The iterative combinations are:
    - normalize_summary: True/False
    - weight_mode: 'equal'/'iterative'
    - weight_scope: 'curve'/'point'
    
    Excludes (weight_mode='equal', weight_scope='point') as redundant,
    yielding 6 iterative combinations.
    
    When ``include_fgpr=True``, two additional FGPR combinations are
    appended (normalized and un-normalized).

    When ``include_student_t=True``, two additional Student-t combinations
    are appended (normalized and un-normalized).
    
    Parameters
    ----------
    include_fgpr : bool
        If True, also include FGPR (normalized + un-normalized).
    include_student_t : bool
        If True, also include Student-t robust aggregation
        (normalized + un-normalized).
    
    Returns
    -------
    List[BatchTestingOptions]
        List of testing option combinations.
    """
    options: List[BatchTestingOptions] = []
    
    # 6 iterative combinations
    for normalization_summary in (True, False):
        for weight_mode in ("equal", "iterative"):
            for weight_scope in ("curve", "point"):
                # Skip redundant: equal weights + pointwise = same as equal + curvewise
                if weight_mode == "equal" and weight_scope == "point":
                    continue
                options.append(BatchTestingOptions(
                    normalization_summary=normalization_summary,
                    weight_mode=weight_mode,
                    weight_scope=weight_scope,
                    aggregation_method="iterative",
                ))
    
    # 2 FGPR combinations (normalized / un-normalized)
    if include_fgpr:
        for normalization_summary in (True, False):
            options.append(BatchTestingOptions(
                normalization_summary=normalization_summary,
                weight_mode="equal",
                weight_scope="curve",
                aggregation_method="fgpr",
            ))
    
    # 2 Student-t combinations (normalized / un-normalized)
    if include_student_t:
        for normalization_summary in (True, False):
            options.append(BatchTestingOptions(
                normalization_summary=normalization_summary,
                weight_mode="equal",
                weight_scope="curve",
                aggregation_method="student_t",
            ))
    
    return options


def discover_groupkeys(
    input_dir: Path,
    pattern: str = "Individual_GPR_*.csv",
    primary_grouping_key: Optional[str] = None,
) -> List[str]:
    """
    Discover available group_keys from Individual_GPR CSV files.
    
    Reads the group_flags metadata from CSV headers to extract grouping values.
    Falls back to filename parsing if metadata is not available.
    
    Parameters
    ----------
    input_dir : Path
        Directory containing individual GPR CSVs.
    pattern : str
        Glob pattern for matching files.
    primary_grouping_key : str, optional
        The primary key to look for in group_flags (e.g., 'potential', 'pH').
        If None, uses the first key found in group_flags.
        
    Returns
    -------
    List[str]
        Sorted list of group key values as strings.
    """
    import json
    import re
    
    group_values = set()
    
    for f in glob.glob(str(input_dir / pattern)):
        filepath = Path(f)
        
        # Try to read from CSV metadata header
        value_found = False
        try:
            with open(filepath, 'r') as file:
                for line in file:
                    if line.startswith('# group_flags:'):
                        # Parse the JSON from the group_flags line
                        json_str = line.split(':', 1)[1].strip()
                        group_flags = json.loads(json_str)
                        
                        # Get value using primary_grouping_key, or first key if not specified
                        if primary_grouping_key and primary_grouping_key in group_flags:
                            group_values.add(str(group_flags[primary_grouping_key]))
                            value_found = True
                        elif group_flags:
                            # Use first key if no primary specified
                            first_key = next(iter(group_flags))
                            group_values.add(str(group_flags[first_key]))
                            value_found = True
                        break
                    elif not line.startswith('#'):
                        # End of metadata header
                        break
        except (json.JSONDecodeError, IOError):
            pass
        
        # Fallback: try to extract from filename (e.g., -1.95V_...)
        if not value_found:
            filename = filepath.stem
            # Match pattern like -1.95V or +1.5V at start of sample_id part
            match = re.search(r'Individual_GPR_([+-]?\d+\.?\d*)V', filename)
            if match:
                group_values.add(match.group(1))
    
    # Try to sort numerically
    try:
        return sorted(group_values, key=float)
    except ValueError:
        return sorted(group_values)


def copy_artifacts(
    source_dir: Path,
    dest_dir: Path,
    patterns: Optional[List[str]] = None,
    verbose: bool = True,
) -> int:
    """
    Copy generated artifacts from source to destination directory.
    
    Parameters
    ----------
    source_dir : Path
        Source directory containing artifacts.
    dest_dir : Path
        Destination directory.
    patterns : List[str], optional
        Glob patterns for files to copy. Defaults to ARTIFACT_PATTERNS.
    verbose : bool
        Print progress.
        
    Returns
    -------
    int
        Number of files copied.
    """
    if patterns is None:
        patterns = ARTIFACT_PATTERNS
    
    os.makedirs(dest_dir, exist_ok=True)
    copied = 0
    
    for pattern in patterns:
        for f in glob.glob(str(source_dir / pattern)):
            shutil.copy2(f, dest_dir)
            copied += 1
    
    if verbose and copied > 0:
        print(f"    Copied {copied} artifacts to {dest_dir}")
    
    return copied


def apply_options_to_config(
    options: BatchTestingOptions,
    summary_gpr_config,
):
    """
    Apply batch testing options to a SummaryGPRConfig.
    
    Parameters
    ----------
    options : BatchTestingOptions
        Testing options to apply.
    summary_gpr_config : SummaryGPRConfig
        Base configuration (will be replaced, not mutated).
        
    Returns
    -------
    SummaryGPRConfig
        New configuration with options applied.
    """
    from dataclasses import replace
    
    updates = dict(
        normalization_summary=options.normalization_summary,
        weight_mode=options.weight_mode,
        weight_scope=options.weight_scope,
    )
    # Operator fusion is never run via the batch grid — disable explicitly
    # so it doesn't leak from the base config loaded from settings.json.
    updates["enable_operator_fusion"] = False

    if options.aggregation_method == "fgpr":
        updates["enable_fgpr"] = True
        updates["enable_student_t"] = False
    elif options.aggregation_method == "student_t":
        updates["enable_fgpr"] = False
        updates["enable_student_t"] = True
    else:
        updates["enable_fgpr"] = False
        updates["enable_student_t"] = False
    return replace(summary_gpr_config, **updates)
