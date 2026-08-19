# pip0_dataloading/data_loader.py
"""
Data Loading Orchestrator for the GPR-FDA Pipeline.

This module provides a unified entry point for all data loading operations,
including finding .cor files, parsing them, and returning structured curve data.

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from .raw_io_config import RawIOCfg
from .cor_file_parser import (
    compute_file_hash,
    find_cor_files,
    parse_cor_file,
)
from .filename_parser import parse_filename


@dataclass
class LoadedCurve:
    """
    Container for a single loaded curve with its metadata.
    
    Attributes
    ----------
    file_path : Path
        Source .cor file path.
    sample_id : str
        Sample identifier from the file.
    group_flags : Dict[str, Any]
        Generic grouping flags (e.g., {"potential": -1.95, "pH": 1.48}).
        Used for grouping curves and building output filenames.
        The primary grouping key is configurable (default: first key in grouping_keys).
    data_points : List[Dict[str, float]]
        List of data point dictionaries with keys:
        'E (Volts)', 'I (A/cm2)', 'T (Seconds)'.
    metadata : Dict[str, Any]
        Additional metadata extracted from filename (e.g., electrolyte, additive, alloy).
        NOT used for grouping or filename generation.
    """
    file_path: Path
    sample_id: str
    group_flags: Dict[str, Any]
    data_points: List[Dict[str, float]]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Cached numpy arrays (computed on first access)
    _x_raw_cache: Optional[Any] = field(default=None, repr=False)
    _y_raw_cache: Optional[Any] = field(default=None, repr=False)
    
    def get_primary_value(self, primary_key: Optional[str] = None) -> Optional[Any]:
        """Get the value for the primary grouping key."""
        if primary_key:
            return self.group_flags.get(primary_key)
        # Return first key's value if no primary specified
        if self.group_flags:
            return next(iter(self.group_flags.values()))
        return None
    
    @property
    def x_raw(self) -> Any:  # np.ndarray, but avoid import
        """Original x values (time) as numpy array."""
        if self._x_raw_cache is None:
            import numpy as np
            self._x_raw_cache = np.array([p['T (Seconds)'] for p in self.data_points])
        return self._x_raw_cache
    
    @property
    def y_raw(self) -> Any:  # np.ndarray, but avoid import
        """Original y values (current) as numpy array."""
        if self._y_raw_cache is None:
            import numpy as np
            self._y_raw_cache = np.array([p['I (A/cm2)'] for p in self.data_points])
        return self._y_raw_cache
    
    @property
    def num_points(self) -> int:
        """Number of data points in the curve."""
        return len(self.data_points)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format (for backward compatibility)."""
        return {
            'file_path': self.file_path,
            'sample_id': self.sample_id,
            'group_flags': self.group_flags,
            'metadata': self.metadata,
            'data_points': self.data_points,
        }


@dataclass
class DataLoadingResult:
    """
    Result container for the data loading operation.
    
    Attributes
    ----------
    curves : List[LoadedCurve]
        All successfully loaded curves.
    curves_by_group : Dict[str, List[LoadedCurve]]
        Curves grouped by a string representation of group_flags.
    curves_by_primary_key : Dict[Any, List[LoadedCurve]]
        Curves grouped by primary grouping key value.
    primary_grouping_key : str
        Name of the primary grouping key used for grouping.
    num_files_processed : int
        Number of .cor files processed.
    num_duplicates_skipped : int
        Number of duplicate files skipped.
    errors : List[str]
        Any error messages encountered during loading.
    """
    curves: List[LoadedCurve] = field(default_factory=list)
    curves_by_group: Dict[str, List[LoadedCurve]] = field(default_factory=dict)
    curves_by_primary_key: Dict[Any, List[LoadedCurve]] = field(default_factory=dict)
    primary_grouping_key: str = ""
    num_files_processed: int = 0
    num_duplicates_skipped: int = 0
    errors: List[str] = field(default_factory=list)
    
    @property
    def num_curves(self) -> int:
        """Total number of curves loaded."""
        return len(self.curves)
    
    @property
    def primary_key_values(self) -> List[Any]:
        """List of unique values for the primary grouping key."""
        return sorted(self.curves_by_primary_key.keys())
    
    @property
    def groups(self) -> List[str]:
        """List of unique group keys."""
        return sorted(self.curves_by_group.keys())
    
    @property
    def num_groups(self) -> int:
        """Number of unique primary key values."""
        return len(self.curves_by_primary_key)
    
    def get_curves_for_primary_key(self, value: Any) -> List[LoadedCurve]:
        """Get all curves for a specific primary key value."""
        return self.curves_by_primary_key.get(value, [])
    
    def summary(self) -> str:
        """Return a summary string of the loading result."""
        key_name = self.primary_grouping_key or "primary key"
        lines = [
            f"Data Loading Summary:",
            f"  Files processed: {self.num_files_processed}",
            f"  Duplicates skipped: {self.num_duplicates_skipped}",
            f"  Total curves loaded: {self.num_curves}",
            f"  Unique {key_name} values: {self.num_groups}",
        ]
        if self.primary_key_values:
            lines.append(f"  {key_name} values: {self.primary_key_values}")
        for val in self.primary_key_values:
            n = len(self.curves_by_primary_key[val])
            lines.append(f"    {val}: {n} curves")
        if self.errors:
            lines.append(f"  Errors: {len(self.errors)}")
        return "\n".join(lines)


class DataLoader:
    """
    Orchestrator for loading electrochemical transient data from .cor files.
    
    This class provides a unified interface for:
    - Finding .cor files in a directory (with duplicate detection)
    - Parsing files to extract curve data
    - Extracting grouping keys from filenames (via configurable token parsing)
    - Grouping curves by configurable primary key
    
    Example
    -------
    >>> from py_FDA_GPR_modules.pip0_dataloading import DataLoader
    >>> loader = DataLoader(path_to_folder="/path/to/data")
    >>> result = loader.load_all()
    >>> print(result.summary())
    >>> for curve in result.get_curves_for_primary_key(-1.95):
    ...     print(curve.sample_id, curve.num_points)
    
    Parameters
    ----------
    path_to_folder : Path or str
        Root directory containing .cor files.
    output_subdir : str, optional
        Subdirectory for output files (default: "output_directory").
    grouping_round_digits : int, optional
        Decimal places for rounding numeric grouping key values (default: 2).
    verbose : bool, optional
        Whether to print progress messages (default: True).
    filename_parsing_config : Dict[str, Any], optional
        Configuration for extracting grouping keys from filenames.
        If provided, grouping keys will be extracted from filenames first,
        with fallback to file content if configured.
        Get this from SettingsManager.get_filename_parsing_config().
    """
    
    def __init__(
        self,
        path_to_folder: Path | str,
        output_subdir: str = "output_directory",
        grouping_round_digits: int = 2,
        verbose: bool = True,
        filename_parsing_config: Optional[Dict[str, Any]] = None,
    ):
        self.path_to_folder = Path(path_to_folder)
        self.output_subdir = output_subdir
        self.grouping_round_digits = grouping_round_digits
        self.verbose = verbose
        self.filename_parsing_config = filename_parsing_config
        
        # Determine primary grouping key from config (fallback to first key)
        self.primary_grouping_key = self._get_primary_grouping_key()
        
        # Create the RawIOCfg for compatibility with existing functions
        self._raw_cfg = RawIOCfg(
            path_to_your_folder=self.path_to_folder,
            output_subdir=self.output_subdir,
        )
    
    def _get_primary_grouping_key(self) -> str:
        """Determine the primary grouping key from configuration."""
        if self.filename_parsing_config:
            # Check for explicit primary_grouping_key
            primary = self.filename_parsing_config.get("primary_grouping_key")
            if primary:
                return primary
            # Fall back to first grouping key
            grouping_keys = self.filename_parsing_config.get("grouping_keys", [])
            if grouping_keys:
                return grouping_keys[0].get("name", "")
        return ""
    
    @property
    def output_directory(self) -> Path:
        """Output directory path."""
        return self._raw_cfg.output_directory
    
    @property
    def raw_io_config(self) -> RawIOCfg:
        """Get the underlying RawIOCfg for backward compatibility."""
        return self._raw_cfg
    
    def find_files(self) -> List[Path]:
        """
        Find all unique .cor files in the directory.
        
        Returns
        -------
        List[Path]
            List of unique .cor file paths.
        """
        return find_cor_files(self._raw_cfg)
    
    def parse_file(self, file_path: Path | str) -> List[LoadedCurve]:
        """
        Parse a single .cor file and return loaded curves.
        
        Grouping keys are extracted in this priority:
        1. From filename (if filename_parsing_config is provided)
        2. From file content (fallback if filename parsing fails or not configured)
        
        Parameters
        ----------
        file_path : Path or str
            Path to the .cor file.
            
        Returns
        -------
        List[LoadedCurve]
            List of curves extracted from the file.
        """
        file_path = Path(file_path)
        raw_curves = parse_cor_file(file_path)
        
        # Try to extract grouping keys and metadata from filename
        filename_group_flags = {}
        filename_metadata = {}
        filename_parsing_success = False
        
        if self.filename_parsing_config:
            parsed = parse_filename(file_path, self.filename_parsing_config)
            filename_parsing_success = parsed.get("_parsing_success", False)
            
            if filename_parsing_success:
                # Extract ONLY grouping keys for group_flags (used in filenames)
                grouping_keys = self.filename_parsing_config.get("grouping_keys", [])
                for key_def in grouping_keys:
                    key_name = key_def.get("name")
                    if key_name and key_name in parsed and parsed[key_name] is not None:
                        filename_group_flags[key_name] = parsed[key_name]
                
                # Extract metadata keys separately (NOT used in filenames)
                metadata_keys = self.filename_parsing_config.get("metadata_keys", [])
                for key_def in metadata_keys:
                    key_name = key_def.get("name")
                    if key_name and key_name in parsed and parsed[key_name] is not None:
                        filename_metadata[key_name] = parsed[key_name]
                
                if self.verbose and filename_group_flags:
                    print(f"  Extracted from filename: {filename_group_flags}")
        
        # Build loaded curves
        loaded_curves = []
        for c in raw_curves:
            # Start with filename-extracted flags
            group_flags = dict(filename_group_flags)
            
            # Determine if we should use file content as fallback
            fallback_to_content = self.filename_parsing_config.get(
                "fallback_to_file_content", True
            ) if self.filename_parsing_config else True
            
            # Get primary grouping key name from config
            primary_key = self.primary_grouping_key
            
            # If primary key not from filename, try file content as fallback
            # File content may have values parsed from the file (e.g., 'potential' from .cor files)
            if primary_key and (primary_key not in group_flags or group_flags[primary_key] is None):
                if fallback_to_content:
                    # Get value from file content using primary_key
                    content_value = c.get(primary_key) 
                    if content_value is not None:
                        group_flags[primary_key] = content_value
            
            loaded_curves.append(LoadedCurve(
                file_path=Path(c['file_path']),
                sample_id=c['sample_id'],
                group_flags=group_flags,
                data_points=c['data_points'],
                metadata=filename_metadata,
            ))
        
        return loaded_curves
    
    @staticmethod
    def _build_group_key(group_flags: Dict[str, Any]) -> str:
        """Build a consistent string key from group flags for dictionary indexing."""
        sorted_items = sorted(group_flags.items())
        return "|".join(f"{k}={v}" for k, v in sorted_items)
    
    def load_all(
        self,
        filter_primary_values: Optional[List[Any]] = None,
    ) -> DataLoadingResult:
        """
        Load all curves from .cor files in the directory.
        
        Parameters
        ----------
        filter_primary_values : List[Any], optional
            If provided, only load curves whose primary key value is in this list.
            
        Returns
        -------
        DataLoadingResult
            Container with all loaded curves and metadata.
        """
        result = DataLoadingResult()
        result.primary_grouping_key = self.primary_grouping_key
        curves_by_primary_key: Dict[Any, List[LoadedCurve]] = defaultdict(list)
        curves_by_group: Dict[str, List[LoadedCurve]] = defaultdict(list)
        
        # Find all .cor files
        cor_files = self.find_files()
        result.num_files_processed = len(cor_files)
        
        if not cor_files:
            if self.verbose:
                print("No .cor files found.")
            return result
        
        if self.verbose:
            print(f"Processing {len(cor_files)} .cor files...")
        
        # Parse each file
        for file_path in cor_files:
            try:
                curves = self.parse_file(file_path)
                for curve in curves:
                    # Get primary key value from group_flags
                    primary_value = curve.group_flags.get(self.primary_grouping_key)
                    
                    # Round if numeric
                    if isinstance(primary_value, (int, float)):
                        primary_value = round(primary_value, self.grouping_round_digits)
                    
                    # Apply filter if specified
                    if filter_primary_values is not None:
                        if primary_value not in filter_primary_values:
                            continue
                    
                    result.curves.append(curve)
                    curves_by_primary_key[primary_value].append(curve)
                    
                    # Also group by full group_flags
                    group_key = self._build_group_key(curve.group_flags)
                    curves_by_group[group_key].append(curve)
                    
            except Exception as e:
                error_msg = f"Error parsing {file_path}: {e}"
                result.errors.append(error_msg)
                if self.verbose:
                    print(error_msg)
        
        result.curves_by_primary_key = dict(curves_by_primary_key)
        result.curves_by_group = dict(curves_by_group)
        
        if self.verbose:
            print(result.summary())
        
        return result
    
    def load_as_dicts(
        self,
        filter_primary_values: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load all curves and return as list of dictionaries.
        
        This method provides backward compatibility with the original
        parse_cor_file return format.
        
        Parameters
        ----------
        filter_primary_values : List[Any], optional
            If provided, only load curves whose primary key value is in this list.
            
        Returns
        -------
        List[Dict]
            List of curve dictionaries in the original format.
        """
        result = self.load_all(filter_primary_values=filter_primary_values)
        return [curve.to_dict() for curve in result.curves]
    
    @classmethod
    def load_from_subdirectories(
        cls,
        parent_folder: Path | str,
        subdirectories: Optional[List[Path | str]] = None,
        filter_primary_values: Optional[List[Any]] = None,
        verbose: bool = True,
        filename_parsing_config: Optional[Dict[str, Any]] = None,
    ) -> DataLoadingResult:
        """
        Load and combine curves from multiple subdirectories into a single dataset.
        
        This is useful when the same experimental dataset is split across
        multiple folders (e.g., different experimental dates) but should be
        analyzed together.
        
        Parameters
        ----------
        parent_folder : Path or str
            Parent directory containing subdirectories with .cor files.
            Used as reference for output naming.
        subdirectories : List[Path or str], optional
            Specific subdirectories to load from. If None, all subdirectories
            in parent_folder will be used.
        filter_primary_values : List[Any], optional
            If provided, only load curves whose primary key value is in this list.
        verbose : bool, optional
            Whether to print progress messages (default: True).
        filename_parsing_config : Dict[str, Any], optional
            Configuration for extracting grouping keys from filenames.
            
        Returns
        -------
        DataLoadingResult
            Combined result with all curves from all subdirectories.
            
        Example
        -------
        >>> from py_FDA_GPR_modules.pip0_dataloading import DataLoader
        >>> 
        >>> # Load from all subdirectories
        >>> result = DataLoader.load_from_subdirectories(
        ...     parent_folder="/path/to/CCNF_CTAB_PT",
        ...     verbose=True,
        ... )
        >>> print(f"Loaded {result.num_curves} curves from {result.num_groups} groups")
        >>> 
        >>> # Load from specific subdirectories
        >>> result = DataLoader.load_from_subdirectories(
        ...     parent_folder="/path/to/CCNF_CTAB_PT",
        ...     subdirectories=["20230616 FeNiCoCu acidB-BWR6", "20230613 FeNiCoCu acidB-BWR5"],
        ... )
        """
        parent_folder = Path(parent_folder)
        
        # Determine subdirectories to process
        if subdirectories is None:
            subdirs = sorted([d for d in parent_folder.iterdir() if d.is_dir()])
        else:
            subdirs = [Path(parent_folder / sd) if not Path(sd).is_absolute() else Path(sd) 
                       for sd in subdirectories]
        
        if verbose:
            print(f"Loading from {len(subdirs)} subdirectories...")
        
        # Determine primary grouping key from config
        primary_grouping_key = ""
        if filename_parsing_config:
            primary = filename_parsing_config.get("primary_grouping_key")
            if primary:
                primary_grouping_key = primary
            else:
                grouping_keys = filename_parsing_config.get("grouping_keys", [])
                if grouping_keys:
                    primary_grouping_key = grouping_keys[0].get("name", "")
        
        # Initialize combined result
        combined_result = DataLoadingResult()
        combined_result.primary_grouping_key = primary_grouping_key
        curves_by_primary_key: Dict[Any, List[LoadedCurve]] = defaultdict(list)
        curves_by_group: Dict[str, List[LoadedCurve]] = defaultdict(list)
        
        total_files = 0
        total_duplicates = 0
        
        for subdir in subdirs:
            if not subdir.exists():
                if verbose:
                    print(f"  WARNING: Subdirectory does not exist: {subdir}")
                continue
            
            # Create loader for this subdirectory
            loader = cls(
                path_to_folder=subdir,
                verbose=False,  # Reduce noise
                filename_parsing_config=filename_parsing_config,
            )
            
            # Load curves from this subdirectory
            sub_result = loader.load_all(filter_primary_values=filter_primary_values)
            
            if verbose:
                print(f"  {subdir.name}: {sub_result.num_curves} curves, "
                      f"groups: {sub_result.primary_key_values}")
            
            # Merge into combined result
            combined_result.curves.extend(sub_result.curves)
            total_files += sub_result.num_files_processed
            total_duplicates += sub_result.num_duplicates_skipped
            combined_result.errors.extend(sub_result.errors)
            
            # Merge groupings
            for curve in sub_result.curves:
                primary_value = curve.group_flags.get(primary_grouping_key)
                if isinstance(primary_value, (int, float)):
                    primary_value = round(primary_value, 2)
                curves_by_primary_key[primary_value].append(curve)
                
                group_key = cls._build_group_key(curve.group_flags)
                curves_by_group[group_key].append(curve)
        
        combined_result.curves_by_primary_key = dict(curves_by_primary_key)
        combined_result.curves_by_group = dict(curves_by_group)
        combined_result.num_files_processed = total_files
        combined_result.num_duplicates_skipped = total_duplicates
        
        if verbose:
            print(f"\n  Total loaded: {combined_result.num_curves} curves")
            print(f"  Combined groups: {combined_result.primary_key_values}")
        
        return combined_result


# Convenience function for quick loading
def load_data(
    path_to_folder: Path | str,
    filter_primary_values: Optional[List[Any]] = None,
    verbose: bool = True,
    filename_parsing_config: Optional[Dict[str, Any]] = None,
) -> DataLoadingResult:
    """
    Convenience function to load all data from a directory.
    
    Parameters
    ----------
    path_to_folder : Path or str
        Root directory containing .cor files.
    filter_primary_values : List[Any], optional
        If provided, only load curves whose primary key value is in this list.
    verbose : bool, optional
        Whether to print progress messages (default: True).
    filename_parsing_config : Dict[str, Any], optional
        Configuration for extracting grouping keys from filenames.
        Get this from SettingsManager.get_filename_parsing_config().
        
    Returns
    -------
    DataLoadingResult
        Container with all loaded curves and metadata.
        
    Example
    -------
    >>> from py_FDA_GPR_modules.pip0_dataloading import load_data, SettingsManager
    >>> 
    >>> # Without filename parsing (uses file content)
    >>> result = load_data("/path/to/data")
    >>> 
    >>> # With filename parsing
    >>> manager = SettingsManager.from_input_folder("/path/to/data")
    >>> parsing_config = manager.get_filename_parsing_config()
    >>> result = load_data("/path/to/data", filename_parsing_config=parsing_config)
    >>> print(f"Loaded {result.num_curves} curves")
    """
    loader = DataLoader(
        path_to_folder=path_to_folder,
        verbose=verbose,
        filename_parsing_config=filename_parsing_config,
    )
    return loader.load_all(filter_primary_values=filter_primary_values)
