# pip0_dataloading/cor_file_parser.py
"""
CorrWare .cor file parser for potentiostatic transient data.

This module provides functions to:
- Find and de-duplicate .cor files by MD5 hash
- Parse .cor files to extract experiment metadata and time-series data

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

import os
import re
import hashlib
from pathlib import Path
from typing import List, Tuple, Optional

from .raw_io_config import RawIOCfg


def compute_file_hash(file_path: str) -> str:
    """Compute the MD5 hash of a file's content to detect duplicates."""
    hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()


def find_cor_files(raw_cfg: RawIOCfg) -> List[Path]:
    """
    Recursively find all .cor files and exclude duplicates.
    
    Parameters
    ----------
    raw_cfg : RawIOCfg
        Configuration object containing the root directory path.
        
    Returns
    -------
    List[Path]
        List of unique .cor file paths.
    """
    root_directory = raw_cfg.path_to_your_folder
    cor_files = []
    seen_hashes = set()
    for dirpath, dirnames, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.lower().endswith('.cor'):
                file_path = os.path.join(dirpath, filename)
                file_hash = compute_file_hash(file_path)
                if file_hash not in seen_hashes:
                    cor_files.append(file_path)
                    seen_hashes.add(file_hash)
                else:
                    print(f"Duplicate file detected and skipped: {file_path}")
    print(f"Found {len(cor_files)} unique .cor files.")
    return cor_files


def parse_cor_file(file_path: Path) -> List[dict]:
    """
    Parse a .cor file to extract potentials and data points.
    
    Parameters
    ----------
    file_path : Path
        Path to the .cor file.
        
    Returns
    -------
    List[dict]
        List of curve dictionaries, each containing:
        - 'file_path': source file path
        - 'sample_id': sample identifier
        - 'potential': applied potential (V)
        - 'data_points': list of {E, I, T} dictionaries
    """
    curves = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if 'Begin Experiment:' in line and 'Potentiostatic' in line:
            curve_data, i = extract_curve_data(lines, i, file_path)
            if curve_data:
                curves.append(curve_data)
        else:
            i += 1
    return curves


def extract_curve_data(lines: List[str], start_index: int, file_path) -> Tuple[Optional[dict], int]:
    """
    Extract data for a single curve starting from start_index.
    
    Parameters
    ----------
    lines : List[str]
        All lines from the .cor file.
    start_index : int
        Line index where the experiment block begins.
    file_path : Path
        Source file path for metadata.
        
    Returns
    -------
    Tuple[Optional[dict], int]
        Tuple of (curve_data dict or None, ending line index).
    """
    potential = None
    data_points = []
    sample_id = None
    data_start_index = None
    i = start_index
    j = i  # Initialize j to handle case where loop doesn't execute
    
    for j in range(i + 1, min(i + 100, len(lines))):
        line_j = lines[j].strip()
        if 'Potential:' in line_j:
            try:
                potential = float(line_j.split(':', 1)[1].strip())
            except ValueError:
                print(f"Could not parse potential at line {j}: {line_j}")
                potential = None
        elif 'Pstat Title:' in line_j:
            sample_id = line_j.split(':', 1)[1].strip()
        elif line_j.startswith('Data Points:'):
            data_start_index = find_data_start(lines, j, str(file_path))
            break
    
    if data_start_index is None:
        print(f"No 'Data Points:' section or 'End Comments' found in file {file_path}. Skipping.")
        return None, j
    
    data_points, end_index = read_data_points(lines, data_start_index)
    
    if potential is not None and data_points:
        curve_data = {
            'file_path': file_path,
            'sample_id': sample_id if sample_id else f"{os.path.basename(file_path)}_curve_{start_index}",
            'potential': potential,
            'data_points': data_points
        }
        return curve_data, end_index
    else:
        print(f"No valid data found for potential in file {file_path}.")
        return None, end_index


def find_data_start(lines: List[str], index: int, file_path: str) -> Optional[int]:
    """
    Find the index where data points start after 'End Comments' section.
    
    Parameters
    ----------
    lines : List[str]
        All lines from the .cor file.
    index : int
        Line index to start searching from.
    file_path : str
        File path for error messages.
        
    Returns
    -------
    Optional[int]
        Line index where data starts, or None if not found.
    """
    for k in range(index + 1, len(lines)):
        line_k = lines[k].strip()
        if line_k == 'End Comments':
            return k + 1
    print(f"'End Comments' not found in file {file_path}. Skipping.")
    return None


def read_data_points(lines: List[str], start_index: int) -> Tuple[List[dict], int]:
    """
    Read data points starting from start_index.
    
    Parameters
    ----------
    lines : List[str]
        All lines from the .cor file.
    start_index : int
        Line index where data points begin.
        
    Returns
    -------
    Tuple[List[dict], int]
        Tuple of (list of data point dicts, ending line index).
        Each data point dict contains 'E (Volts)', 'I (A/cm2)', 'T (Seconds)'.
    """
    data_points = []
    k = start_index  # Initialize k
    for k in range(start_index, len(lines)):
        data_line = lines[k].strip()
        if data_line == '' or 'End' in data_line:
            break
        values = re.split(r'[\t\s]+', data_line)
        if len(values) >= 3:
            try:
                e_volts = float(values[0])
                i_current_density = float(values[1])
                t_seconds = float(values[2])
                data_points.append({
                    'E (Volts)': e_volts,
                    'I (A/cm2)': i_current_density,
                    'T (Seconds)': t_seconds
                })
            except ValueError:
                print(f"Could not parse data line at line {k}: {data_line}")
                continue
        else:
            print(f"Data line doesn't have enough values at line {k}: {data_line}")
            continue
    return data_points, k
