# pip4_efficiency_eval/mc_sampling.py
"""
Monte Carlo sampling utilities for Efficiency Evaluation.

Provides functions for building balanced test sets and determining
the number of MC repeats based on combinatorics.

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import numpy as np

from .efficiency_config import GlobalParams


def bounded_comb(n: int, k: int, *, cap: int) -> int:
    """
    Return C(n, k) capped at cap+1.
    
    Parameters
    ----------
    n : int
        Total items.
    k : int
        Items to choose.
    cap : int
        Maximum value before capping.
        
    Returns
    -------
    int
        min(C(n, k), cap + 1)
    """
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    if k == 0:
        return 1

    result: int = 1
    for i in range(1, k + 1):
        result = (result * (n - k + i)) // i
        if result > cap:
            return cap + 1
    return result


def mc_repeats(n_total: int, m: int, *, globpara: GlobalParams) -> int:
    """
    Number of Monte-Carlo repeats for subset size m drawn from n_total.
    
    If combinatorics are small enough, enumerate all; otherwise use base_repeats.
    
    Parameters
    ----------
    n_total : int
        Total number of curves available.
    m : int
        Subset size.
    globpara : GlobalParams
        Global parameters including base_repeats and max_enum.
        
    Returns
    -------
    int
        Number of repeats to use.
    """
    c = bounded_comb(n_total, m, cap=globpara.max_enum)
    return c if c <= globpara.max_enum else globpara.base_repeats


def balanced_subset(
    idx: np.ndarray,
    occ: np.ndarray,
    m: int,
    rng: np.random.Generator
) -> np.ndarray:
    """
    Return m indices whose occurrence counts are currently minimal.
    
    Ensures balanced sampling across curves by preferring indices
    that have been sampled less frequently.
    
    Parameters
    ----------
    idx : np.ndarray
        Array of all available indices.
    occ : np.ndarray
        Occurrence counts for each index (how many times it has been sampled).
    m : int
        Number of indices to select.
    rng : np.random.Generator
        Random number generator for shuffling.
        
    Returns
    -------
    np.ndarray
        Array of m selected indices.
    """
    sel = []
    remaining = m
    counts_sorted = np.unique(occ)
    for c in counts_sorted:
        cand = idx[occ == c]
        if cand.size == 0:
            continue
        rng.shuffle(cand)
        take = min(remaining, cand.size)
        sel.extend(cand[:take])
        remaining -= take
        if remaining == 0:
            break
    return np.array(sel, dtype=int)


def generate_balanced_subsets(
    n_total: int,
    subset_size: int,
    n_repeats: int,
    *,
    random_seed: int = 42,
) -> list[np.ndarray]:
    """
    Generate multiple balanced subsets for Monte Carlo sampling.
    
    Parameters
    ----------
    n_total : int
        Total number of items to sample from.
    subset_size : int
        Size of each subset.
    n_repeats : int
        Number of subsets to generate.
    random_seed : int
        Random seed for reproducibility.
        
    Returns
    -------
    list[np.ndarray]
        List of subset index arrays.
    """
    idx = np.arange(n_total)
    occ = np.zeros(n_total, dtype=int)
    rng = np.random.default_rng(random_seed)
    
    subsets = []
    for _ in range(n_repeats):
        sel = balanced_subset(idx, occ, subset_size, rng)
        occ[sel] += 1
        subsets.append(sel)
    
    return subsets
