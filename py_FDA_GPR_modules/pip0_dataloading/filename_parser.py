# pip0_dataloading/filename_parser.py
"""
Filename Parser for Extracting Grouping Keys and Metadata.

This module provides utilities to parse filenames based on configurable
token patterns and extract grouping keys and metadata values.

Example filename:
    -1.95V_pH1.48_Na2SO4_H3Cit_NiFeCoCu_acidB(BWR6)_Au-4R2_CTAB_diaphram_20230616.cor

With delimiter="_", tokens become:
    [0] -1.95V
    [1] pH1.48
    [2] Na2SO4
    ...

Grouping keys (e.g., potential, pH) are extracted using token_index and optional regex.

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union


# =============================================================================
# Title Formatting Utilities
# =============================================================================

def format_group_key_title(group_key: str) -> str:
    """
    Format a group_key string for use in plot titles.
    
    Converts formats like:
    - "potential=-1.95" -> "potential = -1.95"
    - "pH=7.4|temp=25" -> "pH = 7.4, temp = 25"
    - "pH=1.48|potential=-1.95" -> "pH = 1.48, potential = -1.95"
    - "-1.95" -> "Group = -1.95"
    - "" or "unknown" -> "Unknown Group"
    
    Handles multiple group keys separated by '|'.
    
    Parameters
    ----------
    group_key : str
        Group key string from aggregation (e.g., "potential=-1.95" or 
        "pH=1.48|potential=-1.95" for multi-key groupings).
        
    Returns
    -------
    str
        Human-readable formatted string for plot titles.
    """
    if not group_key or group_key == "unknown":
        return "Unknown Group"
    
    # Handle old-style numeric-only keys (for backward compat with potential values)
    try:
        float(group_key)
        return f"Group = {group_key}"
    except ValueError:
        pass
    
    # Split by | for multi-key groups (e.g., "pH=1.48|potential=-1.95")
    parts = group_key.split('|')
    formatted_parts = []
    
    for part in parts:
        if '=' in part:
            # Format "key=value" -> "key = value"
            key, value = part.split('=', 1)
            formatted_parts.append(f"{key} = {value}")
        else:
            # Keep as-is if no = separator
            formatted_parts.append(part)
    
    return ', '.join(formatted_parts)


# =============================================================================
# Filename Parsing Functions
# =============================================================================

def parse_filename(
    filename: Union[str, Path],
    parsing_config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Parse a filename to extract grouping keys and metadata.
    
    Parameters
    ----------
    filename : str or Path
        The filename (with or without path) to parse.
    parsing_config : dict
        Configuration dictionary with keys:
        - delimiter: str (e.g., "_")
        - grouping_keys: List[dict] with {name, token_index, dtype, regex_extract}
        - metadata_keys: List[dict] with {name, token_index, dtype, regex_extract}
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with extracted values:
        - All grouping_keys as key-value pairs
        - All metadata_keys as key-value pairs
        - "_parsing_success": bool indicating if parsing succeeded
        - "_failed_keys": list of keys that failed to parse
    """
    # Get just the filename without path
    if isinstance(filename, Path):
        fname = filename.name
    else:
        fname = Path(filename).name
    
    delimiter = parsing_config.get("delimiter", "_")
    grouping_keys = parsing_config.get("grouping_keys", [])
    metadata_keys = parsing_config.get("metadata_keys", [])
    
    # Split filename into tokens
    tokens = fname.split(delimiter)
    
    result = {
        "_parsing_success": True,
        "_failed_keys": [],
        "_tokens": tokens,
    }
    
    # Extract all keys
    all_keys = grouping_keys + metadata_keys
    for key_def in all_keys:
        name = key_def.get("name")
        token_index = key_def.get("token_index", 0)
        dtype = key_def.get("dtype", "string")
        regex_pattern = key_def.get("regex_extract")
        
        if name is None:
            continue
        
        # Get the token (handle negative indices)
        try:
            token = tokens[token_index]
        except IndexError:
            result[name] = None
            result["_failed_keys"].append(name)
            result["_parsing_success"] = False
            continue
        
        # Extract value using regex if provided
        value_str = token
        if regex_pattern:
            match = re.search(regex_pattern, token)
            if match and match.groups():
                value_str = match.group(1)
            elif match:
                value_str = match.group(0)
            else:
                # Regex didn't match
                result[name] = None
                result["_failed_keys"].append(name)
                result["_parsing_success"] = False
                continue
        
        # Convert to appropriate dtype
        try:
            if dtype == "float":
                result[name] = float(value_str)
            elif dtype == "int":
                result[name] = int(value_str)
            else:  # string
                result[name] = value_str.strip()
        except (ValueError, TypeError):
            result[name] = None
            result["_failed_keys"].append(name)
            result["_parsing_success"] = False
    
    return result


def extract_grouping_value(
    filename: Union[str, Path],
    parsing_config: Dict[str, Any],
    key_name: Optional[str] = None,
) -> Optional[Any]:
    """
    Extract a specific grouping value from a filename.
    
    Parameters
    ----------
    filename : str or Path
        The filename to parse.
    parsing_config : dict
        Parsing configuration from settings.
    key_name : str, optional
        Name of the grouping key to extract. 
        If None, uses primary_grouping_key from config.
        
    Returns
    -------
    Any
        The extracted value, or None if extraction failed.
    """
    if key_name is None:
        key_name = parsing_config.get("primary_grouping_key", "potential")
    
    parsed = parse_filename(filename, parsing_config)
    return parsed.get(key_name)


def batch_parse_filenames(
    file_paths: List[Union[str, Path]],
    parsing_config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Parse multiple filenames and return extracted metadata for each.
    
    Parameters
    ----------
    file_paths : List[str or Path]
        List of file paths to parse.
    parsing_config : dict
        Parsing configuration from settings.
        
    Returns
    -------
    List[Dict[str, Any]]
        List of parsed metadata dictionaries, one per file.
        Each dict includes 'file_path' key with the original path.
    """
    results = []
    for fp in file_paths:
        parsed = parse_filename(fp, parsing_config)
        parsed["file_path"] = str(fp)
        results.append(parsed)
    return results


def get_unique_grouping_values(
    file_paths: List[Union[str, Path]],
    parsing_config: Dict[str, Any],
    key_name: Optional[str] = None,
    round_digits: Optional[int] = None,
) -> List[Any]:
    """
    Get unique grouping values from a list of filenames.
    
    Parameters
    ----------
    file_paths : List[str or Path]
        List of file paths to parse.
    parsing_config : dict
        Parsing configuration from settings.
    key_name : str, optional
        Grouping key name. Defaults to primary_grouping_key.
    round_digits : int, optional
        If provided, round numeric values to this many decimal places.
        
    Returns
    -------
    List[Any]
        Sorted list of unique grouping values.
    """
    if key_name is None:
        key_name = parsing_config.get("primary_grouping_key", "potential")
    
    values = set()
    for fp in file_paths:
        val = extract_grouping_value(fp, parsing_config, key_name)
        if val is not None:
            if round_digits is not None and isinstance(val, float):
                val = round(val, round_digits)
            values.add(val)
    
    # Sort: numeric values first, then strings
    try:
        return sorted(values)
    except TypeError:
        # Mixed types - convert all to strings for sorting
        return sorted(values, key=str)


def group_files_by_key(
    file_paths: List[Union[str, Path]],
    parsing_config: Dict[str, Any],
    key_name: Optional[str] = None,
    round_digits: Optional[int] = None,
) -> Dict[Any, List[Path]]:
    """
    Group files by a grouping key value.
    
    Parameters
    ----------
    file_paths : List[str or Path]
        List of file paths to group.
    parsing_config : dict
        Parsing configuration from settings.
    key_name : str, optional
        Grouping key name. Defaults to primary_grouping_key.
    round_digits : int, optional
        If provided, round numeric values before grouping.
        
    Returns
    -------
    Dict[Any, List[Path]]
        Dictionary mapping grouping values to lists of file paths.
    """
    if key_name is None:
        key_name = parsing_config.get("primary_grouping_key", "potential")
    
    groups = {}
    for fp in file_paths:
        fp = Path(fp)
        val = extract_grouping_value(fp, parsing_config, key_name)
        
        if val is None:
            # Put in "unknown" group
            val = "_unknown"
        elif round_digits is not None and isinstance(val, float):
            val = round(val, round_digits)
        
        if val not in groups:
            groups[val] = []
        groups[val].append(fp)
    
    return groups
