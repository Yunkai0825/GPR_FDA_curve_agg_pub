# pip0_dataloading/raw_io_config.py
"""
Configuration dataclass for raw data I/O paths.

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RawIOCfg:
    """Paths and duplicate filter configuration for raw data loading."""
    path_to_your_folder: Path
    output_subdir: str = "output_directory"

    @property
    def output_directory(self) -> Path:
        return self.path_to_your_folder / self.output_subdir
