# pip0_dataloading/csv_curve_parser.py
"""
Generic CSV curve parser (one curve per file).

This module provides functions to:
- Find and de-duplicate curve files with a given extension (default .csv)
- Parse a CSV file into the same curve-dict format produced by parse_cor_file

Expected file layout:
    # optional provenance lines starting with '#'
    col_a,col_b,col_c,...
    1.0,2.0,3.0,...

Every numeric column is kept in each data-point dict, so pip1 can select its
x/y columns by name (pip1_datapreprocessing.column_names in the settings JSON).
Grouping keys come from the filename (pip0_dataloading.filename_parsing).

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

import os
from pathlib import Path
from typing import List

import pandas as pd

from .raw_io_config import RawIOCfg
from .cor_file_parser import compute_file_hash


def find_curve_files(raw_cfg: RawIOCfg, file_extension: str = ".csv") -> List[Path]:
    """
    Recursively find all files with the given extension and exclude duplicates.

    Parameters
    ----------
    raw_cfg : RawIOCfg
        Configuration object containing the root directory path.
    file_extension : str
        File extension to match (case-insensitive), e.g. ".csv".

    Returns
    -------
    List[Path]
        List of unique file paths.
    """
    root_directory = raw_cfg.path_to_your_folder
    ext = file_extension.lower()
    found = []
    seen_hashes = set()
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in sorted(filenames):
            if filename.lower().endswith(ext):
                file_path = os.path.join(dirpath, filename)
                file_hash = compute_file_hash(file_path)
                if file_hash not in seen_hashes:
                    found.append(file_path)
                    seen_hashes.add(file_hash)
                else:
                    print(f"Duplicate file detected and skipped: {file_path}")
    print(f"Found {len(found)} unique {ext} files.")
    return found


def parse_csv_curve_file(file_path: Path) -> List[dict]:
    """
    Parse a CSV file holding a single curve.

    Parameters
    ----------
    file_path : Path
        Path to the CSV file.

    Returns
    -------
    List[dict]
        One curve dictionary (same keys as parse_cor_file output):
        - 'file_path': source file path
        - 'sample_id': file stem
        - 'data_points': list of {column_name: value} dictionaries
    """
    df = pd.read_csv(file_path, comment="#")
    df = df.apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all").dropna(axis=0, how="any")
    if df.empty:
        print(f"No numeric data found in file {file_path}. Skipping.")
        return []
    return [{
        'file_path': file_path,
        'sample_id': Path(file_path).stem,
        'data_points': df.to_dict(orient="records"),
    }]
