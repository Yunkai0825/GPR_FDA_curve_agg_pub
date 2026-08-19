# pip4_efficiency_eval/pip4_efficiency_eval_orchestrator.py
"""
High-level orchestrator for Efficiency Evaluation.

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Any, TYPE_CHECKING

import numpy as np
import pandas as pd

from .efficiency_config import DirParams, GlobalParams, ScaleParams, PlotParams
from .efficiency_core import (
    LearningCurveResult,
    process_potential_learning_curve,
    summarize_learning_curve,
)
from .efficiency_plotting import (
    plot_learning_curve_from_detailed,
    plot_iteration_statistics_from_detailed,
    aggregate_detailed_to_summary,
    plot_sigma_btw_comparison,
    plot_covariance_heatmaps,
    plot_covariance_diagonal,
    plot_pointwise_sigma_btw,
    export_sigma_btw_csv,
)

# Import loader from pip3
from ..pip3_FDA_scoring_and_aggregations import (
    IndividualGPRData,
    load_all_individual_gprs,
    group_gprs_by_key,
)


class EfficiencyOrchestrator:
    """
    Orchestrator for data efficiency evaluation.
    
    Example
    -------
    >>> from py_FDA_GPR_modules.pip4_efficiency_eval import EfficiencyOrchestrator
    >>> from py_FDA_GPR_modules.pip3_FDA_scoring_and_aggregations import SummaryGPRConfig, SummaryGPRHyperParams
    >>> 
    >>> dirpara = DirParams(indiv_dir=Path("output/"))
    >>> orchestrator = EfficiencyOrchestrator(
    ...     dirpara=dirpara,
    ...     summary_gpr_config=SummaryGPRConfig(),
    ...     summary_gpr_hyperparams=SummaryGPRHyperParams(),
    ... )
    >>> results = orchestrator.process_all()
    """
    
    def __init__(
        self,
        dirpara: DirParams,
        summary_gpr_config,
        summary_gpr_hyperparams,
        *,
        globpara: Optional[GlobalParams] = None,
        scapara: Optional[ScaleParams] = None,
        plotpara: Optional[PlotParams] = None,
        verbose: bool = True,
        aggregation_method: str = "iterative",
    ):
        """
        Initialize orchestrator.
        
        Parameters
        ----------
        dirpara : DirParams
            Directory parameters.
        summary_gpr_config : SummaryGPRConfig
            Configuration for summary GPR algorithm.
        summary_gpr_hyperparams : SummaryGPRHyperParams
            Hyperparameters for summary GPR.
        globpara : GlobalParams, optional
            Global experiment parameters.
        scapara : ScaleParams, optional
            Scaling parameters.
        plotpara : PlotParams, optional
            Plot parameters.
        verbose : bool
            Print progress.
        aggregation_method : str
            "iterative" for variance-based methods, "fgpr" for functional GPR.
        """
        self.dirpara = dirpara
        self.summary_gpr_config = summary_gpr_config
        self.summary_gpr_hyperparams = summary_gpr_hyperparams
        self.globpara = globpara or GlobalParams()
        self.scapara = scapara or ScaleParams()
        self.plotpara = plotpara or PlotParams()
        self.verbose = verbose
        self.aggregation_method = aggregation_method
        
        # Ensure output directory exists
        if self.dirpara.output_dir:
            os.makedirs(self.dirpara.output_dir, exist_ok=True)
    
    def _log(self, msg: str):
        """Print message if verbose."""
        if self.verbose:
            print(msg)
    
    def process_group(
        self,
        group_key: str,
        gpr_list: List[IndividualGPRData],
    ) -> Optional[LearningCurveResult]:
        """
        Process one group of curves.
        
        Parameters
        ----------
        group_key : str
            Group key string (e.g., "potential=-1.95" or "-1.95").
        gpr_list : List[IndividualGPRData]
            List of IndividualGPRData objects for this group.
            
        Returns
        -------
        LearningCurveResult or None
            Learning curve results (summary + detailed), or None if not enough curves.
        """
        N = len(gpr_list)
        self._log(f"\n  Processing group {group_key} ({N} curves)...")
        
        # Make filename-safe by replacing problematic characters
        filename_key = group_key.replace('|', '_').replace('=', '_')
        csv_stem = f"LearningCurve_{filename_key}"

        # Define plot callback for layer-by-layer interim plots
        config_info = {
            "subtitle": (
                f"Weight mode: {self.summary_gpr_config.weight_mode}, "
                f"scope: {self.summary_gpr_config.weight_scope}"
            )
        }

        def _interim_plot_cb(df_detailed, layer):
            """Generate interim learning-curve plot after a layer completes."""
            plot_learning_curve_from_detailed(
                df_detailed,
                filename_key,
                self.dirpara.output_dir,
                scapara=self.scapara,
                plotpara=self.plotpara,
                summary_config_info=config_info,
                verbose=False,
            )

        # Call core processing function (layered mode for real-time CSV + resume)
        lc_result = process_potential_learning_curve(
            gpr_list,
            summary_gpr_config=self.summary_gpr_config,
            summary_gpr_hyperparams=self.summary_gpr_hyperparams,
            globpara=self.globpara,
            scapara=self.scapara,
            verbose=self.verbose,
            aggregation_method=self.aggregation_method,
            layered=True,
            output_dir=str(self.dirpara.output_dir),
            csv_stem=csv_stem,
            plot_callback=_interim_plot_cb,
        )
        
        if lc_result is None:
            self._log(f"  Group {group_key}: skipped (not enough curves)")
            return None
        
        # Save CSVs
        if self.dirpara.output_dir:
            # Save summary learning curve CSV
            summary_path = self.dirpara.output_dir / f"LearningCurve_{filename_key}_summary.csv"
            lc_result.summary.to_csv(summary_path, index=False)
            self._log(f"    -> Saved summary: {summary_path}")
            
            # Save detailed individual runs CSV
            detailed_path = self.dirpara.output_dir / f"LearningCurve_{filename_key}_detailed.csv"
            lc_result.detailed.to_csv(detailed_path, index=False)
            self._log(f"    -> Saved detailed: {detailed_path}")
            
            # Save wide summary (transposed)
            summarize_learning_curve(
                lc_result.summary,
                csv_stem=f"LearningCurve_{filename_key}",
                out_dir=str(self.dirpara.output_dir),
            )
            
            # Plot learning curve (from detailed data)
            config_info = {
                "subtitle": (
                    f"Weight mode: {self.summary_gpr_config.weight_mode}, "
                    f"scope: {self.summary_gpr_config.weight_scope}"
                )
            }
            plot_learning_curve_from_detailed(
                lc_result.detailed,
                filename_key,
                self.dirpara.output_dir,
                scapara=self.scapara,
                plotpara=self.plotpara,
                summary_config_info=config_info,
                verbose=self.verbose,
            )
            
            # Plot iteration statistics (from detailed data)
            plot_iteration_statistics_from_detailed(
                lc_result.detailed,
                filename_key,
                self.dirpara.output_dir,
                verbose=self.verbose,
            )
            
            # Save covariance matrices (FGPR) as .npy and .csv files
            if lc_result.cov_matrices:
                cov_dir = self.dirpara.output_dir / "cov_matrices"
                os.makedirs(cov_dir, exist_ok=True)
                for ss, cov in lc_result.cov_matrices.items():
                    npy_path = cov_dir / f"Cagg_{filename_key}_ss{ss}.npy"
                    np.save(npy_path, cov)
                    csv_path = cov_dir / f"Cagg_{filename_key}_ss{ss}.csv"
                    np.savetxt(csv_path, cov, delimiter=',')
                self._log(f"    -> Saved {len(lc_result.cov_matrices)} covariance "
                          f"matrices (.npy + .csv) to {cov_dir}")
            
            # Save pointwise sigma_btw arrays (iterative methods) as .npy + csv
            if lc_result.sigma_btw_pointwise_arrays:
                sbtw_dir = self.dirpara.output_dir / "sigma_btw_pointwise"
                os.makedirs(sbtw_dir, exist_ok=True)
                for ss, arr in lc_result.sigma_btw_pointwise_arrays.items():
                    npy_path = sbtw_dir / f"sigma_btw_pw_{filename_key}_ss{ss}.npy"
                    np.save(npy_path, arr)
                    csv_path = sbtw_dir / f"sigma_btw_pw_{filename_key}_ss{ss}.csv"
                    np.savetxt(csv_path, arr, delimiter=',',
                               header='sigma_btw_pointwise_obs_scale', comments='')
                self._log(f"    -> Saved {len(lc_result.sigma_btw_pointwise_arrays)} "
                          f"sigma_btw_pointwise arrays to {sbtw_dir}")
            
            # ----- Default sigma_btw diagnostics (always generated) -----
            self._generate_sigma_btw_diagnostics(
                lc_result, filename_key, group_key,
            )
        
        return lc_result
    
    def _generate_sigma_btw_diagnostics(
        self,
        lc_result: LearningCurveResult,
        filename_key: str,
        group_key: str,
    ) -> None:
        """
        Generate between-model variance plots and CSV by default.
        
        For every method run (FGPR or iterative), this produces:
        - sigma_btw vs subset_size line plot  (PNG)
        - sigma_btw summary statistics table  (CSV)
        - FGPR: covariance heatmaps + diagonal std profile
        - Iterative: pointwise sigma_btw profiles
        """
        out = self.dirpara.output_dir
        if out is None:
            return
        
        # Build a single-method dict {label: detailed_df} for plotting
        method_label = (
            f"{self.summary_gpr_config.weight_mode}_"
            f"{self.summary_gpr_config.weight_scope}"
        )
        if self.aggregation_method.lower() == "fgpr":
            method_label = "fgpr"
            if self.summary_gpr_config.normalization_summary:
                method_label += "_normalised"
            else:
                method_label += "_observation"
        
        method_det = {method_label: lc_result.detailed}
        
        # 1. sigma_btw vs subset_size plot
        if "sigma_btw" in lc_result.detailed.columns:
            plot_sigma_btw_comparison(
                method_det, group_key,
                out / f"SigmaBtw_{method_label}_{filename_key}.png",
                plotpara=self.plotpara,
                verbose=self.verbose,
            )
        
        # 2. sigma_btw summary CSV
        export_sigma_btw_csv(
            method_det,
            out / f"sigma_btw_{method_label}_{filename_key}.csv",
            verbose=self.verbose,
        )
        
        # 3. FGPR-specific: covariance heatmaps + diagonal std profile
        if lc_result.cov_matrices:
            cov_dir = out / "cov_matrices"
            plot_covariance_heatmaps(
                lc_result.cov_matrices, group_key,
                cov_dir / f"CovHeatmap_{filename_key}.png",
                plotpara=self.plotpara,
                verbose=self.verbose,
            )
            plot_covariance_diagonal(
                lc_result.cov_matrices, group_key,
                cov_dir / f"CovDiagStd_{filename_key}.png",
                plotpara=self.plotpara,
                verbose=self.verbose,
            )
        
        # 4. Iterative-specific: pointwise sigma_btw profiles
        if lc_result.sigma_btw_pointwise_arrays:
            sbtw_dir = out / "sigma_btw_pointwise"
            plot_pointwise_sigma_btw(
                lc_result.sigma_btw_pointwise_arrays,
                method_label, group_key,
                sbtw_dir / f"SigmaBtw_pointwise_{filename_key}.png",
                plotpara=self.plotpara,
                verbose=self.verbose,
            )
    
    def process_all(self) -> Dict[str, LearningCurveResult]:
        """
        Process all groups in the input directory.
        
        Returns
        -------
        Dict[str, LearningCurveResult]
            Results keyed by group_key, each containing summary and detailed DataFrames.
        """
        self._log(f"Loading individual curves from: {self.dirpara.indiv_dir}")
        
        # Use pip3 loader to load all GPRs
        all_gprs = load_all_individual_gprs(
            directory=str(self.dirpara.indiv_dir),
            pattern="Individual_GPR_*.csv",
            verbose=self.verbose,
        )
        
        if not all_gprs:
            self._log("  No curves found!")
            return {}
        
        # Group by full group_key (e.g., "pH=1.48|potential=-1.95")
        gprs_by_group = group_gprs_by_key(all_gprs, key_attr="group_key")
        
        self._log(f"Found {len(gprs_by_group)} groups: {sorted(gprs_by_group.keys())}")
        
        results: Dict[str, LearningCurveResult] = {}
        
        for group_key in sorted(gprs_by_group.keys()):
            gpr_list = gprs_by_group[group_key]
            
            result = self.process_group(group_key, gpr_list)
            if result is not None:
                results[group_key] = result
        
        self._log(f"\nEfficiency evaluation complete: {len(results)} groups processed")
        
        return results
