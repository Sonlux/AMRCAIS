"""
Walk-Forward Backtesting Harness for AMRCAIS (Phase 3.3).

Implements anchored and rolling walk-forward validation for regime
prediction models.  This is the gold-standard method for evaluating
in-sample vs out-of-sample performance and detecting overfitting.

Walk-Forward Methodology:
    ┌───────────────┬──────┐───────────────┬──────┐───────────────┬──────┐
    │  Train (IS₁)  │ OOS₁ │  Train (IS₂)  │ OOS₂ │  Train (IS₃)  │ OOS₃ │
    └───────────────┴──────┘───────────────┴──────┘───────────────┴──────┘
    ↑ Fixed or rolling start

    IS  = In-Sample  (train window)
    OOS = Out-of-Sample (test window)

Each window produces performance metrics that are then aggregated
across all folds to give robust estimates of model quality.

Classes:
    WalkForwardConfig: Configuration for walk-forward runs.
    FoldResult: Results for a single train/test fold.
    WalkForwardResult: Aggregated walk-forward results.
    WalkForwardHarness: Main harness that orchestrates validation.

Example:
    >>> harness = WalkForwardHarness(config=WalkForwardConfig(n_folds=6))
    >>> result = harness.run(data, model_fn, predict_fn)
    >>> print(f"OOS R²: {result.oos_r_squared:.4f}")
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation.

    Attributes:
        n_folds: Number of walk-forward folds (windows).
        train_ratio: Fraction of available data used for each training window.
        min_train_size: Minimum training window in trading days.
        test_size: Out-of-sample window in trading days.
        anchored: If True, training start is fixed (anchored).
                  If False, training window rolls forward (rolling).
        gap_days: Days gap between train and test (avoid lookahead).
        purge_days: Extra purge period for contaminated samples.
    """

    n_folds: int = 6
    train_ratio: float = 0.7
    min_train_size: int = 252  # ~1 year
    test_size: int = 63  # ~3 months
    anchored: bool = True
    gap_days: int = 1
    purge_days: int = 5


@dataclass
class FoldResult:
    """Results for a single walk-forward fold.

    Attributes:
        fold_index: Zero-based fold number.
        train_start: Training window start date.
        train_end: Training window end date.
        test_start: Test window start date.
        test_end: Test window end date.
        train_samples: Number of training samples.
        test_samples: Number of test samples.
        in_sample_r2: In-sample R² (goodness of fit).
        oos_r2: Out-of-sample R² (predictive power).
        in_sample_accuracy: In-sample classification accuracy.
        oos_accuracy: Out-of-sample classification accuracy.
        in_sample_mse: In-sample mean squared error.
        oos_mse: Out-of-sample mean squared error.
        train_time_seconds: Time to train in this fold.
        metadata: Extra metrics or diagnostics.
    """

    fold_index: int = 0
    train_start: str = ""
    train_end: str = ""
    test_start: str = ""
    test_end: str = ""
    train_samples: int = 0
    test_samples: int = 0
    in_sample_r2: float = 0.0
    oos_r2: float = 0.0
    in_sample_accuracy: float = 0.0
    oos_accuracy: float = 0.0
    in_sample_mse: float = 0.0
    oos_mse: float = 0.0
    train_time_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize fold result."""
        return {
            "fold_index": self.fold_index,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "train_samples": self.train_samples,
            "test_samples": self.test_samples,
            "in_sample_r2": round(self.in_sample_r2, 6),
            "oos_r2": round(self.oos_r2, 6),
            "in_sample_accuracy": round(self.in_sample_accuracy, 4),
            "oos_accuracy": round(self.oos_accuracy, 4),
            "in_sample_mse": round(self.in_sample_mse, 6),
            "oos_mse": round(self.oos_mse, 6),
            "train_time_seconds": round(self.train_time_seconds, 2),
            "metadata": self.metadata,
        }


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward validation results.

    Attributes:
        config: Configuration used.
        folds: Per-fold results.
        mean_oos_r2: Average out-of-sample R².
        std_oos_r2: Std dev of OOS R².
        mean_oos_accuracy: Average OOS accuracy.
        mean_is_r2: Average in-sample R².
        overfit_ratio: Ratio of IS to OOS R² (>2 suggests overfit).
        total_time_seconds: Total validation time.
        model_name: Name of the model validated.
    """

    config: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    folds: List[FoldResult] = field(default_factory=list)
    mean_oos_r2: float = 0.0
    std_oos_r2: float = 0.0
    mean_oos_accuracy: float = 0.0
    mean_is_r2: float = 0.0
    overfit_ratio: float = 0.0
    total_time_seconds: float = 0.0
    model_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize full results."""
        return {
            "model_name": self.model_name,
            "n_folds": len(self.folds),
            "anchored": self.config.anchored,
            "mean_oos_r2": round(self.mean_oos_r2, 6),
            "std_oos_r2": round(self.std_oos_r2, 6),
            "mean_oos_accuracy": round(self.mean_oos_accuracy, 4),
            "mean_is_r2": round(self.mean_is_r2, 6),
            "overfit_ratio": round(self.overfit_ratio, 4),
            "total_time_seconds": round(self.total_time_seconds, 2),
            "folds": [f.to_dict() for f in self.folds],
            "verdict": self._verdict(),
        }

    def _verdict(self) -> str:
        """Human-readable verdict on model quality."""
        if self.mean_oos_r2 < 0:
            return "FAIL — negative OOS R² (worse than naïve)"
        if self.overfit_ratio > 3.0:
            return "WARNING — severe overfitting detected"
        if self.overfit_ratio > 2.0:
            return "CAUTION — moderate overfitting"
        if self.mean_oos_r2 > 0.05:
            return "PASS — meaningful predictive signal"
        return "MARGINAL — weak but positive signal"


class WalkForwardHarness:
    """Orchestrates walk-forward validation for AMRCAIS models.

    The harness splits time-series data into sequential train/test
    folds, fits the model on train data, evaluates on test data,
    and aggregates results across folds.

    Args:
        config: Walk-forward configuration.

    Example:
        >>> harness = WalkForwardHarness()
        >>> result = harness.run(
        ...     data=market_data,
        ...     model_fn=lambda train: model.fit(train),
        ...     predict_fn=lambda model, test: model.predict(test),
        ...     target_col="Regime",
        ... )
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None) -> None:
        self._config = config or WalkForwardConfig()

    def run(
        self,
        data: pd.DataFrame,
        model_fn: Callable[[pd.DataFrame], Any],
        predict_fn: Callable[[Any, pd.DataFrame], np.ndarray],
        target_col: str = "target",
        feature_cols: Optional[List[str]] = None,
        model_name: str = "unknown",
    ) -> WalkForwardResult:
        """Run walk-forward validation.

        Args:
            data: Full dataset with DatetimeIndex.
            model_fn: Callable(train_df) → fitted model object.
            predict_fn: Callable(model, test_df) → predictions array.
            target_col: Name of the target column.
            feature_cols: Feature columns (default: all except target).
            model_name: Descriptive model name.

        Returns:
            WalkForwardResult with per-fold metrics and aggregates.
        """
        logger.info(
            f"Starting walk-forward validation: model={model_name}, "
            f"folds={self._config.n_folds}, anchored={self._config.anchored}"
        )

        start_time = time.time()

        if feature_cols is None:
            feature_cols = [c for c in data.columns if c != target_col]

        # Generate fold indices
        fold_specs = self._generate_folds(len(data))

        folds: List[FoldResult] = []

        for i, (train_start, train_end, test_start, test_end) in enumerate(fold_specs):
            logger.info(
                f"  Fold {i + 1}/{len(fold_specs)}: "
                f"train[{train_start}:{train_end}] → test[{test_start}:{test_end}]"
            )

            train_data = data.iloc[train_start:train_end]
            test_data = data.iloc[test_start:test_end]

            if len(train_data) < self._config.min_train_size:
                logger.warning(f"  Fold {i + 1} skipped: insufficient training data")
                continue

            if len(test_data) < 5:
                logger.warning(f"  Fold {i + 1} skipped: insufficient test data")
                continue

            fold_result = self._run_fold(
                fold_index=i,
                train_data=train_data,
                test_data=test_data,
                model_fn=model_fn,
                predict_fn=predict_fn,
                target_col=target_col,
                feature_cols=feature_cols,
            )
            folds.append(fold_result)

        # Aggregate results
        result = self._aggregate(folds, model_name, time.time() - start_time)

        logger.info(
            f"Walk-forward complete: "
            f"OOS R²={result.mean_oos_r2:.4f} ± {result.std_oos_r2:.4f}, "
            f"overfit_ratio={result.overfit_ratio:.2f}, "
            f"verdict={result.to_dict()['verdict']}"
        )

        return result

    def _generate_folds(self, n_samples: int) -> List[Tuple[int, int, int, int]]:
        """Generate train/test fold index boundaries.

        Args:
            n_samples: Total number of data points.

        Returns:
            List of (train_start, train_end, test_start, test_end) tuples.
        """
        cfg = self._config
        test_size = cfg.test_size
        gap = cfg.gap_days + cfg.purge_days

        folds = []

        if cfg.anchored:
            # Anchored: training always starts at index 0
            first_test_start = cfg.min_train_size + gap
            available = n_samples - first_test_start
            step = max(1, available // cfg.n_folds)

            for i in range(cfg.n_folds):
                test_start = first_test_start + i * step
                test_end = min(test_start + test_size, n_samples)
                train_end = test_start - gap

                if train_end <= 0 or test_start >= n_samples:
                    break

                folds.append((0, train_end, test_start, test_end))
        else:
            # Rolling: training window slides forward
            total_window = int(n_samples * cfg.train_ratio) + test_size + gap
            step = max(1, (n_samples - total_window) // max(1, cfg.n_folds - 1))

            for i in range(cfg.n_folds):
                window_start = i * step
                train_size = int((total_window - test_size - gap))
                train_end = window_start + train_size
                test_start = train_end + gap
                test_end = min(test_start + test_size, n_samples)

                if test_end > n_samples or train_size < cfg.min_train_size:
                    break

                folds.append((window_start, train_end, test_start, test_end))

        return folds

    def _run_fold(
        self,
        fold_index: int,
        train_data: pd.DataFrame,
        test_data: pd.DataFrame,
        model_fn: Callable,
        predict_fn: Callable,
        target_col: str,
        feature_cols: List[str],
    ) -> FoldResult:
        """Execute a single walk-forward fold.

        Args:
            fold_index: Fold number.
            train_data: Training DataFrame.
            test_data: Testing DataFrame.
            model_fn: Model fitting function.
            predict_fn: Prediction function.
            target_col: Target column name.
            feature_cols: Feature column names.

        Returns:
            FoldResult with metrics.
        """
        t0 = time.time()

        try:
            # Train
            model = model_fn(train_data)

            # In-sample predictions
            is_preds = predict_fn(model, train_data)
            is_targets = train_data[target_col].values

            # Out-of-sample predictions
            oos_preds = predict_fn(model, test_data)
            oos_targets = test_data[target_col].values

            train_time = time.time() - t0

            # Compute metrics
            dates = (
                str(train_data.index[0]),
                str(train_data.index[-1]),
                str(test_data.index[0]),
                str(test_data.index[-1]),
            )

            return FoldResult(
                fold_index=fold_index,
                train_start=dates[0],
                train_end=dates[1],
                test_start=dates[2],
                test_end=dates[3],
                train_samples=len(train_data),
                test_samples=len(test_data),
                in_sample_r2=self._r_squared(is_targets, is_preds),
                oos_r2=self._r_squared(oos_targets, oos_preds),
                in_sample_accuracy=self._accuracy(is_targets, is_preds),
                oos_accuracy=self._accuracy(oos_targets, oos_preds),
                in_sample_mse=self._mse(is_targets, is_preds),
                oos_mse=self._mse(oos_targets, oos_preds),
                train_time_seconds=train_time,
            )

        except Exception as exc:
            logger.error(f"Fold {fold_index} failed: {exc}")
            return FoldResult(
                fold_index=fold_index,
                train_samples=len(train_data),
                test_samples=len(test_data),
                train_time_seconds=time.time() - t0,
                metadata={"error": str(exc)},
            )

    def _aggregate(
        self,
        folds: List[FoldResult],
        model_name: str,
        total_time: float,
    ) -> WalkForwardResult:
        """Aggregate fold results into WalkForwardResult.

        Args:
            folds: List of per-fold results.
            model_name: Model name.
            total_time: Total elapsed time.

        Returns:
            Aggregated WalkForwardResult.
        """
        if not folds:
            return WalkForwardResult(
                config=self._config,
                model_name=model_name,
                total_time_seconds=total_time,
            )

        oos_r2s = [f.oos_r2 for f in folds]
        is_r2s = [f.in_sample_r2 for f in folds]
        oos_accs = [f.oos_accuracy for f in folds]

        mean_oos = float(np.mean(oos_r2s))
        std_oos = float(np.std(oos_r2s))
        mean_is = float(np.mean(is_r2s))

        overfit = mean_is / mean_oos if mean_oos > 0 else float("inf")

        return WalkForwardResult(
            config=self._config,
            folds=folds,
            mean_oos_r2=mean_oos,
            std_oos_r2=std_oos,
            mean_oos_accuracy=float(np.mean(oos_accs)),
            mean_is_r2=mean_is,
            overfit_ratio=overfit,
            total_time_seconds=total_time,
            model_name=model_name,
        )

    # ── Metric Helpers ────────────────────────────────────────

    @staticmethod
    def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute R² (coefficient of determination).

        Args:
            y_true: Actual values.
            y_pred: Predicted values.

        Returns:
            R² score (-inf to 1.0).
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

        if ss_tot == 0:
            return 0.0

        return float(1.0 - ss_res / ss_tot)

    @staticmethod
    def _mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute mean squared error.

        Args:
            y_true: Actual values.
            y_pred: Predicted values.

        Returns:
            MSE value.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.mean((y_true - y_pred) ** 2))

    @staticmethod
    def _accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute classification accuracy (for discrete predictions).

        For continuous predictions, discretizes by rounding.

        Args:
            y_true: True labels.
            y_pred: Predicted labels.

        Returns:
            Accuracy in [0, 1].
        """
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Discretize continuous predictions
        if y_pred.dtype in (np.float32, np.float64):
            y_pred = np.round(y_pred).astype(int)
        if y_true.dtype in (np.float32, np.float64):
            y_true = np.round(y_true).astype(int)

        if len(y_true) == 0:
            return 0.0

        return float(np.mean(y_true == y_pred))
