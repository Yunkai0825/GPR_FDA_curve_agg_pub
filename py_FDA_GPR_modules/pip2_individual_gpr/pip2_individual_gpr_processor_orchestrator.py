# pip2_individual_gpr/gpr_processor.py
"""
Individual GPR Processor Orchestrator for the GPR-FDA Pipeline.

This module provides a unified entry point for fitting GPR models to
preprocessed curves from pip1_datapreprocessing.

The orchestrator delegates to:
- gpr_functions: Core GPR fitting logic
- gpr_utilities: Export and plotting utilities

Author: Yunkai Sun (C-STEEL, CSE, ANL)
"""

from __future__ import annotations

import gc
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

from .gpr_config import GPRCfg, ExportCfg, SharedGridConfig
from .gpr_functions import (
    PosteriorResult,
    perform_gpr,
    generate_predictions,
    validate_gpr,
    refit_with_frozen_hyperparameters,
    regulate_to_shared_grid,
    full_regulation_workflow,
)
from .gpr_utilities import (
    group_curves_by_key,
    export_gpr_result_to_csv,
    export_covariance_matrix,
    export_diagnostic_data,
    export_validation_results,
    plot_individual_gpr,
)

# Import ScalingInfo for type hints
from py_FDA_GPR_modules.pip1_datapreprocessing import ScalingInfo


@dataclass
class GPRFitResult:
    """
    Result of fitting GPR to a single curve.
    
    Contains predictions in both transformed (normalized) and original scales
    with full posterior distribution for curve aggregation.
    
    Attributes
    ----------
    sample_id : str
        Sample identifier.
    group_flags : Dict[str, Any]
        Generic grouping flags (e.g., {"potential": -1.95}).
    index_id : int
        Unique index for this curve.
    
    Predictions (Transformed Space):
    ---------------------------------
    x_pred_transformed : np.ndarray
        Prediction x points (transformed, e.g., log-time).
    y_pred_transformed : np.ndarray
        Predicted y values (transformed/normalized).
    y_std_transformed : np.ndarray
        Prediction uncertainty (transformed/normalized).
    
    Predictions (Original Space):
    ------------------------------
    x_pred : np.ndarray
        Prediction x points (original scale, e.g., time in seconds).
    y_pred : np.ndarray
        Predicted y values (original scale, e.g., A/cm2).
    y_std : np.ndarray
        Prediction uncertainty (original scale).
    
    Scaling Information:
    --------------------
    x_scaling : ScalingInfo
        Transformation info for x (from pip1 preprocessing).
    y_scaling : ScalingInfo
        Transformation info for y (from pip1 preprocessing).
        Contains physical normalization factor s_r.
    
    Validation Metrics:
    -------------------
    validation_mae : float
        Mean Absolute Error on validation set (transformed space).
    validation_rmse : float
        Root Mean Squared Error on validation set (transformed space).
    
    GPR Model Components:
    ---------------------
    gpr_model : GaussianProcessRegressor
        Fitted GPR model.
    scaler_X : StandardScaler
        Sklearn scaler for X values (statistical normalization).
    scaler_y : StandardScaler
        Sklearn scaler for y values (statistical normalization).
    hyperparams_str : str
        String representation of optimized kernel.
    hyperparams : dict
        Dictionary of kernel hyperparameters.
    
    Posterior Distribution:
    -----------------------
    posterior : PosteriorResult
        Complete posterior distribution with full covariance.
        Contains covariance matrix for curve aggregation.
    validation_metrics_extended : dict, optional
        Extended validation metrics (NLPD, calibration).
    """
    # Identifiers
    sample_id: str
    group_flags: Dict[str, Any]
    index_id: int = 0
    
    # Predictions in transformed space
    x_pred_transformed: Optional[np.ndarray] = None
    y_pred_transformed: Optional[np.ndarray] = None
    y_std_transformed: Optional[np.ndarray] = None
    
    # Predictions in original space
    x_pred: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None
    y_std: Optional[np.ndarray] = None
    
    # Scaling information (from pip1)
    x_scaling: Optional[ScalingInfo] = None
    y_scaling: Optional[ScalingInfo] = None
    
    # Validation metrics
    validation_mae: float = 0.0
    validation_rmse: float = 0.0
    
    # GPR model components
    gpr_model: Optional[GaussianProcessRegressor] = None
    gpr_model_train: Optional[GaussianProcessRegressor] = None  # Training-only GPR (for calibration)
    scaler_X: Optional[StandardScaler] = None
    scaler_y: Optional[StandardScaler] = None
    hyperparams_str: str = ""
    hyperparams: Dict[str, Any] = field(default_factory=dict)
    
    # Validation data (for calibration plots)
    x_val: Optional[np.ndarray] = None
    y_val: Optional[np.ndarray] = None
    
    # Posterior distribution
    posterior: Optional[PosteriorResult] = None
    validation_metrics_extended: Optional[Dict[str, float]] = None
    
    # For curve aggregation in pip3 - store physical scale factor explicitly
    physical_scale_factor: float = 1.0
    
    def get_posterior_covariance(self) -> Optional[np.ndarray]:
        """
        Get posterior covariance matrix for curve aggregation.
        
        Returns covariance in normalized space. For original units,
        multiply by (physical_scale_factor * statistical_std)².
        """
        if self.posterior is None:
            return None
        return self.posterior.covariance
    
    def get_posterior_covariance_diagonal(self) -> np.ndarray:
        """Get variance (diagonal of covariance) as 1D array."""
        if self.posterior is not None:
            return self.posterior.get_covariance_diagonal()
        return self.y_std_transformed ** 2 if self.y_std_transformed is not None else np.array([])
    
    def release_model_memory(self) -> None:
        """Release heavy sklearn objects to free memory after export.

        After the GPR result has been exported to CSV (predictions +
        covariance matrix), the fitted sklearn ``GaussianProcessRegressor``
        objects and full posterior covariance are no longer needed in
        memory.  Calling this method sets them to ``None`` so that the
        garbage collector can reclaim the memory.

        The lightweight fields (predictions, std, scaling info, etc.) are
        retained so that downstream aggregation (pip3) can still load
        results from disk.
        """
        # Drop fitted GPR models (each holds L_ (n,n), K (n,n), alpha_ (n,))
        self.gpr_model = None
        self.gpr_model_train = None
        self.scaler_X = None
        self.scaler_y = None
        # Drop full covariance from posterior (keep mean/std)
        if self.posterior is not None:
            self.posterior.covariance = None
            self.posterior.covariance_cholesky = None
            self.posterior.covariance_sparse = None
        # Drop validation data
        self.x_val = None
        self.y_val = None

    def get_validation_dict(self) -> Dict[str, Any]:
        """Get validation results as dictionary."""
        result = {
            'sample_id': self.sample_id,
            'index_id': self.index_id,
            'group_flags': self.group_flags,
            'MAE': self.validation_mae,
            'RMSE': self.validation_rmse,
            'Optimized Hyperparameters': self.hyperparams_str,
        }
        # Add extended metrics if available
        if self.validation_metrics_extended is not None:
            result.update({k: v for k, v in self.validation_metrics_extended.items()})
        # Add group flags as separate columns
        for k, v in self.group_flags.items():
            result[k] = v
        # Add hyperparameters
        for param_name, param_value in self.hyperparams.items():
            clean_name = param_name.replace('__', '_').replace('k1', 'kernel1').replace('k2', 'kernel2')
            result[clean_name] = param_value
        return result


@dataclass
class GPRProcessingResult:
    """
    Result container for GPR processing operations.
    
    Attributes
    ----------
    results : List[GPRFitResult]
        Successfully fitted GPR results.
    results_by_group : Dict[str, List[GPRFitResult]]
        Results grouped by generic group_flags.
    skipped : List[Dict[str, Any]]
        Skipped curves with reasons.
    config : GPRCfg
        Configuration used.
    """
    results: List[GPRFitResult] = field(default_factory=list)
    results_by_group: Dict[str, List[GPRFitResult]] = field(default_factory=dict)
    skipped: List[Dict[str, Any]] = field(default_factory=list)
    config: Optional[GPRCfg] = None
    
    @property
    def num_results(self) -> int:
        return len(self.results)
    
    @property
    def num_skipped(self) -> int:
        return len(self.skipped)
    
    @property
    def groups(self) -> List[str]:
        return sorted(self.results_by_group.keys())
    
    def get_results_for_group(self, group_key: str) -> List[GPRFitResult]:
        return self.results_by_group.get(group_key, [])
    
    def summary(self) -> str:
        lines = [
            "GPR Processing Summary:",
            f"  Successfully fitted: {self.num_results} curves",
            f"  Skipped: {self.num_skipped} curves",
            f"  Unique groups: {len(self.groups)}",
        ]
        for group_key in self.groups:
            n = len(self.results_by_group[group_key])
            lines.append(f"    {group_key}: {n} curves")

        # Aggregate key hyperparameter ranges across fitted curves for quick inspection
        def _collect_values(keys):
            vals = []
            for r in self.results:
                hp = r.hyperparams or {}
                for k in keys:
                    if k in hp:
                        try:
                            vals.append(float(hp[k]))
                        except Exception:
                            # Skip non-numeric or array-like values that cannot cast cleanly
                            try:
                                vals.append(float(np.asarray(hp[k]).item()))
                            except Exception:
                                pass
                        break
            return vals

        def _add_range(label: str, vals):
            if not vals:
                return
            arr = np.asarray(vals, dtype=float)
            lines.append(f"  {label}: min={np.nanmin(arr):.3g}, max={np.nanmax(arr):.3g}")

        _add_range("sigma_f (amplitude)", _collect_values([
            "k1__k1__constant_value", "k1__constant_value", "constant_value"
        ]))
        _add_range("length_scale", _collect_values([
            "k1__k2__length_scale", "k1__length_scale", "length_scale"
        ]))
        _add_range("nu", _collect_values([
            "k1__k2__nu", "k1__nu", "nu"
        ]))
        _add_range("sigma_m (noise_level)", _collect_values([
            "k2__noise_level", "noise_level"
        ]))
        return "\n".join(lines)


class IndividualGPRProcessor:
    """
    Orchestrator for fitting GPR models to preprocessed curves.
    
    This class coordinates calls to:
    - gpr_functions: For GPR fitting, validation, predictions
    - gpr_utilities: For export and plotting
    
    Example
    -------
    >>> from py_FDA_GPR_modules.pip0_dataloading import DataLoader
    >>> from py_FDA_GPR_modules.pip1_datapreprocessing import DataPreprocessor
    >>> from py_FDA_GPR_modules.pip2_individual_gpr import IndividualGPRProcessor
    >>> 
    >>> # Load and preprocess data
    >>> loader = DataLoader(path_to_folder="/path/to/data")
    >>> preprocessor = DataPreprocessor()
    >>> preprocessed = preprocessor.preprocess_all(loader.load_all().curves)
    >>> 
    >>> # Fit GPR models
    >>> gpr_processor = IndividualGPRProcessor(output_directory="/path/to/output")
    >>> result = gpr_processor.fit_all(preprocessed.curves)
    >>> print(result.summary())
    """
    
    def __init__(
        self,
        gpr_config: Optional[GPRCfg] = None,
        export_config: Optional[ExportCfg] = None,
        output_directory: Optional[Union[Path, str]] = None,
        verbose: bool = True,
    ):
        if gpr_config is None:
            raise ValueError(
                "gpr_config must be provided explicitly (constructed from JSON settings). "
                "No default GPRCfg is allowed — grid parameters must come from JSON."
            )
        self.gpr_config = gpr_config
        self.export_config = export_config or ExportCfg()
        self.output_directory = Path(output_directory) if output_directory else None
        self.verbose = verbose
        self._index_counter = 1
    
    def fit_single(
        self,
        preprocessed_curve: Any,
        X_pred: Optional[np.ndarray] = None,
        X_shared: Optional[np.ndarray] = None,
        index_id: Optional[int] = None,
    ) -> Tuple[Optional[GPRFitResult], Optional[str]]:
        """
        Fit GPR to a single preprocessed curve with optional regulation to shared grid.
        
        Workflow (when shared_grid.enabled and X_shared provided):
        ----------------------------------------------------------
        Step A: Fit GPR on training data to optimize hyperparameters θ*
        Step B: Refit on ALL data (train + val) with frozen θ*
        Step C: Regulate predictions to shared grid X_shared
        
        Parameters
        ----------
        preprocessed_curve : PreprocessedCurve
            Preprocessed curve from pip1.
        X_pred : np.ndarray, optional
            Legacy: Prediction grid (ignored if X_shared is provided).
        X_shared : np.ndarray, optional
            Shared regulation grid J_R. Shape (n_grid,) or (n_grid, 1).
            If provided and shared_grid.enabled, uses full regulation workflow.
        index_id : int, optional
            Curve index.
            
        Returns
        -------
        Tuple[GPRFitResult, str]
            (result, skip_reason) - result is None if skipped.
        """
        cfg = self.gpr_config
        
        if index_id is None:
            index_id = self._index_counter
            self._index_counter += 1
        
        sample_id = preprocessed_curve.sample_id
        group_flags = preprocessed_curve.group_flags
        
        # Get scaling info from preprocessed curve
        x_scaling: Optional[ScalingInfo] = preprocessed_curve.x_scaling
        y_scaling: Optional[ScalingInfo] = preprocessed_curve.y_scaling
        
        # Get training data (already transformed and downsampled)
        x_train = preprocessed_curve.x_train.reshape(-1, 1)
        y_train = preprocessed_curve.y_train
        x_val, y_val = preprocessed_curve.get_validation_data()
        
        # Get physical scaling factor s_r from pip1 normalization
        physical_scale_factor = y_scaling.params.get('factor', 1.0) if y_scaling is not None else 1.0
        
        # Determine if we should use the regulation workflow
        use_regulation = (
            cfg.shared_grid.enabled and 
            X_shared is not None
        )
        
        # Prepare X_shared shape
        if X_shared is not None:
            X_shared = np.asarray(X_shared).reshape(-1, 1)
        
        if use_regulation:
            # =====================================================================
            # REGULATION WORKFLOW (Steps A, B, C)
            # =====================================================================
            
            # Combine all available data for refitting
            if len(x_val) > 0:
                X_all = np.vstack([x_train, x_val.reshape(-1, 1)])
                y_all = np.concatenate([y_train, y_val])
            else:
                X_all = x_train
                y_all = y_train
            
            try:
                # Full regulation: train → refit → regulate
                (
                    posterior,
                    gpr,
                    gpr_train,
                    scaler_X,
                    scaler_y,
                    hyperparams_str,
                    hyperparams,
                ) = full_regulation_workflow(
                    X_train=x_train,
                    y_train=y_train,
                    X_all=X_all,
                    y_all=y_all,
                    X_shared=X_shared,
                    physical_scale_factor=physical_scale_factor,
                    gpr_cfg=cfg,
                )
            except Exception as e:
                return None, f"Regulation workflow failed: {e}"
            
            # Prediction grid is the shared grid
            x_pred_transformed = X_shared
            
            # Validation on the TRAIN-ONLY GPR (θ* tuned on train split)
            # gpr_train shares the same scalers as gpr/gpr_refitted, preventing
            # scaler mismatch in validation/calibration.
            validation_metrics = None
            if len(x_val) > 0 and gpr_train is not None:
                validation_metrics = validate_gpr(x_val, y_val, gpr_train, scaler_X, scaler_y)
                mae = validation_metrics['MAE']
                rmse = validation_metrics['RMSE']
                if self.verbose:
                    msg = f"Sample {sample_id} - MAE: {mae:.4f}, RMSE: {rmse:.4f}"
                    msg += f", NLPD: {validation_metrics.get('NLPD', float('nan')):.4f}"
                    msg += " [regulated]"
                    print(msg)
            else:
                mae, rmse = np.nan, np.nan
                gpr_train = None  # No validation data, no need for train-only GPR
                if self.verbose:
                    print(f"Sample {sample_id} - No validation data [regulated]")
            
        else:
            # =====================================================================
            # LEGACY WORKFLOW (single-step fitting)
            # =====================================================================
            gpr, scaler_X, scaler_y, hyperparams_str, hyperparams = perform_gpr(
                x_train, y_train, gpr_cfg=cfg
            )
            
            # In legacy mode, gpr_train is the same as gpr (no refitting)
            gpr_train = gpr
            
            if gpr is None:
                return None, "GPR fitting failed"
            
            assert scaler_X is not None and scaler_y is not None
            
            # Validation
            validation_metrics = None
            if len(x_val) > 0:
                validation_metrics = validate_gpr(x_val, y_val, gpr, scaler_X, scaler_y)
                mae = validation_metrics['MAE']
                rmse = validation_metrics['RMSE']
                if self.verbose:
                    msg = f"Sample {sample_id} - MAE: {mae:.4f}, RMSE: {rmse:.4f}"
                    msg += f", NLPD: {validation_metrics['NLPD']:.4f}"
                    print(msg)
            else:
                mae, rmse = np.nan, np.nan
                if self.verbose:
                    print(f"Sample {sample_id} - No validation data")
            
            # Build prediction grid (in transformed space)
            if X_pred is None:
                x_min, x_max = x_train.min(), x_train.max()
                x_pred_transformed = np.linspace(x_min, x_max, cfg.num_X_pred_points_individual_default).reshape(-1, 1)
            else:
                x_pred_transformed = X_pred
            
            # Generate predictions
            posterior = generate_predictions(
                gpr, scaler_X, scaler_y, x_pred_transformed,
                physical_scale_factor, gpr_cfg=cfg
            )
        
        # Get predictions in both spaces from posterior
        y_pred_transformed = posterior.mean
        y_std_transformed = posterior.std
        y_pred = posterior.get_mean_original_units()
        y_std = posterior.get_std_original_units()
        
        # =====================================================================
        # Convert x_pred to original space if scaling info available
        # =====================================================================
        x_pred_original: Optional[np.ndarray] = None
        if x_scaling is not None:
            x_pred_original = x_scaling.inverse_transform(x_pred_transformed.flatten())
        
        return GPRFitResult(
            sample_id=sample_id,
            group_flags=group_flags,
            index_id=index_id,
            # Predictions in transformed space
            x_pred_transformed=x_pred_transformed,
            y_pred_transformed=y_pred_transformed,
            y_std_transformed=y_std_transformed,
            # Predictions in original space
            x_pred=x_pred_original,
            y_pred=y_pred,
            y_std=y_std,
            # Scaling info (from pip1)
            x_scaling=x_scaling,
            y_scaling=y_scaling,
            # Validation metrics
            validation_mae=mae,
            validation_rmse=rmse,
            # GPR model components
            gpr_model=gpr,
            gpr_model_train=gpr_train,  # Training-only GPR for calibration
            scaler_X=scaler_X,
            scaler_y=scaler_y,
            hyperparams_str=hyperparams_str or "",
            hyperparams=hyperparams or {},
            # Validation data for calibration plots
            x_val=x_val if len(x_val) > 0 else None,
            y_val=y_val if len(y_val) > 0 else None,
            # Posterior distribution
            posterior=posterior,
            validation_metrics_extended=validation_metrics,
            physical_scale_factor=physical_scale_factor,
        ), None
    
    def fit_all(
        self,
        preprocessed_curves: List[Any],
        group_key_extractor: Optional[Callable[[Any], str]] = None,
        export_results: bool = True,
        plot_results: bool = True,
    ) -> GPRProcessingResult:
        """
        Fit GPR models to all preprocessed curves.
        
        Parameters
        ----------
        preprocessed_curves : List[Any]
            List of PreprocessedCurve objects from pip1.
        group_key_extractor : Callable, optional
            Function to extract grouping key from a curve. If None, uses group_flags.
        export_results : bool
            Whether to export results to CSV.
        plot_results : bool
            Whether to generate plots.
        
        Returns
        -------
        GPRProcessingResult
            Container with all fitted GPR results.
        """
        result = GPRProcessingResult(config=self.gpr_config)
        results_by_group: Dict[str, List[GPRFitResult]] = defaultdict(list)
        
        # Group curves by their group_flags
        # Default grouper builds key from all group_flags
        if group_key_extractor is None:
            group_key_extractor = lambda c: self._build_group_key(c.group_flags)
        
        # Group curves
        curves_by_group: Dict[str, List[Any]] = defaultdict(list)
        for curve in preprocessed_curves:
            key = group_key_extractor(curve)
            curves_by_group[key].append(curve)
        
        if self.verbose:
            print(f"Fitting GPR to {len(preprocessed_curves)} curves across {len(curves_by_group)} groups...")
        
        for group_key, curves in curves_by_group.items():
            if self.verbose:
                print(f"\nProcessing group '{group_key}' ({len(curves)} curves)")
            
            # =====================================================================
            # Compute shared grid J_R for this group
            # =====================================================================
            shared_grid_cfg = self.gpr_config.shared_grid
            
            # Collect all x values from this group for grid computation
            all_x_train = np.concatenate([c.x_train for c in curves])
            all_x_val = []
            for c in curves:
                x_v, _ = c.get_validation_data()
                if len(x_v) > 0:
                    all_x_val.append(x_v)
            if all_x_val:
                all_x_total = np.concatenate([all_x_train] + all_x_val)
            else:
                all_x_total = all_x_train
            
            # Determine number of points
            n_points = (
                self.gpr_config.num_X_pred_points_individual_high
                if len(curves) < self.gpr_config.num_curves_threshold
                else self.gpr_config.num_X_pred_points_individual_default
            )
            
            # Compute shared grid based on method
            X_shared: np.ndarray
            if shared_grid_cfg.enabled and shared_grid_cfg.method == "explicit":
                # Use explicit grid expression
                X_shared = shared_grid_cfg.evaluate_explicit_grid()
                if X_shared is None:
                    raise ValueError("explicit_grid must be specified when method='explicit'")
                if self.verbose:
                    print(f"  Using explicit shared grid: {len(X_shared)} points, "
                          f"range [{X_shared.min():.4f}, {X_shared.max():.4f}]")
                # Warn if the grid doesn't cover the data range
                data_min = all_x_total.min()
                data_max = all_x_total.max()
                grid_min = X_shared.min()
                grid_max = X_shared.max()
                if grid_max < data_max - 0.01:
                    import warnings
                    warnings.warn(
                        f"Shared grid upper bound ({grid_max:.4f}) is below "
                        f"data max ({data_max:.4f}). GPR predictions will be "
                        f"truncated. Update explicit_grid in settings to cover "
                        f"the full data range.",
                        stacklevel=2,
                    )
                if grid_min > data_min + 0.01:
                    import warnings
                    warnings.warn(
                        f"Shared grid lower bound ({grid_min:.4f}) is above "
                        f"data min ({data_min:.4f}). GPR predictions will be "
                        f"truncated. Update explicit_grid in settings to cover "
                        f"the full data range.",
                        stacklevel=2,
                    )
            elif shared_grid_cfg.enabled and shared_grid_cfg.method == "auto":
                # Compute from ALL curves in the group (train + val data)
                X_shared = shared_grid_cfg.compute_auto_grid(all_x_total, n_points)
                if self.verbose:
                    print(f"  Using auto shared grid: {len(X_shared)} points, "
                          f"range [{X_shared.min():.4f}, {X_shared.max():.4f}]")
            elif shared_grid_cfg.enabled and shared_grid_cfg.method == "per_group":
                # Compute from this group's training data only
                X_shared = shared_grid_cfg.compute_auto_grid(all_x_train, n_points)
                if self.verbose:
                    print(f"  Using per-group shared grid: {len(X_shared)} points, "
                          f"range [{X_shared.min():.4f}, {X_shared.max():.4f}]")
            else:
                # Legacy behavior: use training data range only
                X_shared = np.linspace(all_x_train.min(), all_x_train.max(), n_points)
                if self.verbose:
                    print(f"  Using legacy grid (regulation disabled): {len(X_shared)} points")
            
            X_shared = X_shared.reshape(-1, 1)
            
            validation_results = []
            
            for curve in curves:
                # Pass X_shared to enable regulation workflow
                gpr_result, skip_reason = self.fit_single(
                    curve, 
                    X_pred=X_shared,  # Legacy parameter (used if regulation disabled)
                    X_shared=X_shared if shared_grid_cfg.enabled else None,  # Regulation grid
                )
                
                if gpr_result is not None:
                    result.results.append(gpr_result)
                    results_by_group[group_key].append(gpr_result)
                    
                    validation_results.append(gpr_result.get_validation_dict())
                    
                    # Delegate to gpr_utilities for export (pass full result for metadata)
                    if export_results and self.output_directory:
                        export_gpr_result_to_csv(
                            gpr_result,
                            group_key,
                            self.output_directory,
                            verbose=self.verbose
                        )
                        # Export full covariance matrix in normalized space
                        export_covariance_matrix(
                            gpr_result,
                            self.output_directory,
                            verbose=self.verbose
                        )
                        # Export training/validation diagnostic data
                        # (predictions at train/val points, standardized
                        #  residuals, calibration stats → plots 5-8)
                        export_diagnostic_data(
                            gpr_result,
                            curve.df_downsampled,
                            self.output_directory,
                            x_col='x_transformed',
                            y_col='y_transformed',
                            verbose=self.verbose,
                        )
                    
                    # Delegate to gpr_utilities for diagnostic plotting
                    if plot_results and self.export_config.plot_individual_gpr and self.output_directory:
                        plot_individual_gpr(
                            gpr_result,
                            curve.df_downsampled,
                            self.output_directory,
                            x_col='x_transformed',
                            y_col='y_transformed',
                            x_scaling=gpr_result.x_scaling,
                            y_scaling=gpr_result.y_scaling,
                            export_cfg=self.export_config
                        )
                    
                    # Release heavy model objects to prevent memory
                    # exhaustion when fitting many curves sequentially.
                    # The predictions and covariance have already been
                    # exported to CSV above.
                    if export_results:
                        gpr_result.release_model_memory()
                        gc.collect()
                else:
                    result.skipped.append({
                        'sample_id': curve.sample_id,
                        'group_key': group_key,
                        'group_flags': curve.group_flags,
                        'reason': skip_reason,
                    })
            
            # Delegate to gpr_utilities for validation export
            if export_results and self.output_directory and validation_results:
                export_validation_results(
                    validation_results, group_key, self.output_directory, verbose=self.verbose
                )
        
        result.results_by_group = dict(results_by_group)
        
        if self.verbose:
            print("\n" + result.summary())
        
        return result
    
    @staticmethod
    def _build_group_key(group_flags: Dict[str, Any]) -> str:
        """Build a consistent string key from group flags for dictionary indexing."""
        sorted_items = sorted(group_flags.items())
        return "|".join(f"{k}={v}" for k, v in sorted_items)
    
    def fit_from_preprocessing_result(
        self,
        preprocessing_result: Any,
        export_results: bool = True,
        plot_results: bool = True,
    ) -> GPRProcessingResult:
        """Fit GPR directly from a PreprocessingResult."""
        return self.fit_all(
            preprocessing_result.curves,
            export_results=export_results,
            plot_results=plot_results,
        )


# Convenience function
def fit_individual_gprs(
    preprocessed_curves: List[Any],
    output_directory: Optional[Union[Path, str]] = None,
    gpr_config: Optional[GPRCfg] = None,
    export_results: bool = True,
    plot_results: bool = True,
    verbose: bool = True,
) -> GPRProcessingResult:
    """Convenience function to fit GPR models to preprocessed curves."""
    processor = IndividualGPRProcessor(
        gpr_config=gpr_config,
        output_directory=output_directory,
        verbose=verbose,
    )
    return processor.fit_all(
        preprocessed_curves,
        export_results=export_results,
        plot_results=plot_results,
    )
