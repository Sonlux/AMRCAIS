"""
Factor Exposure Analyzer for AMRCAIS.

Tracks exposure to systematic risk factors (momentum, value, quality, size)
and interprets factor performance in regime context.  Supports both
generic factor columns and **Fama-French / AQR** academic factor models
via multi-factor rolling OLS regression.

Factor premia vary dramatically across regimes:

- Risk-On Growth: Momentum and quality lead
- Risk-Off Crisis: Quality and low-vol outperform, value and momentum suffer
- Stagflation: Commodities and real assets, equities struggle
- Disinflationary Boom: Growth and quality lead, value lags

Classes:
    FactorExposureAnalyzer: Regime-adaptive factor analysis
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import logging

import pandas as pd
import numpy as np

from src.modules.base import AnalyticalModule, ModuleSignal

logger = logging.getLogger(__name__)


@dataclass
class FactorReturn:
    """Single factor return observation.
    
    Attributes:
        factor: Factor name
        return_value: Period return
        z_score: Return relative to historical distribution
        percentile: Historical percentile
    """
    factor: str
    return_value: float
    z_score: float
    percentile: float


@dataclass
class OLSResult:
    """Result of a (multi-factor) rolling OLS regression.

    Attributes:
        alpha: Intercept (annualised excess return not explained by factors).
        betas: Dict mapping factor name → estimated loading / beta.
        r_squared: R² of the regression.
        t_stats: Dict mapping factor name → t-statistic for its beta.
        residual_vol: Standard deviation of residuals (annualised).
        n_obs: Number of observations in the rolling window.
    """
    alpha: float
    betas: Dict[str, float]
    r_squared: float
    t_stats: Dict[str, float] = field(default_factory=dict)
    residual_vol: float = 0.0
    n_obs: int = 0


class FactorExposureAnalyzer(AnalyticalModule):
    """Analyzes factor exposures with regime-adaptive interpretation.
    
    Systematic risk factors (momentum, value, quality, size, volatility)
    have very different expected returns across regimes. This module:
    
    1. Tracks factor returns and z-scores
    2. Interprets factor performance in regime context
    3. Identifies factor rotation signals
    4. Flags factor crowding risks
    
    REGIME-SPECIFIC FACTOR EXPECTATIONS:
    
    Risk-On Growth (1):
        - Momentum: Strong (trends persist)
        - Quality: Moderate
        - Value: Underperforms (growth wins)
        - Size: Small caps outperform
    
    Risk-Off Crisis (2):
        - Momentum: Crashes (reversals)
        - Quality: Outperforms (flight to quality)
        - Value: Variable (can work in recovery)
        - Size: Large caps outperform
    
    Stagflation (3):
        - Momentum: Weak
        - Quality: Moderate
        - Value: Can work (real assets)
        - Commodities: Outperform
    
    Disinflationary Boom (4):
        - Momentum: Strong
        - Quality: Strong
        - Growth: Outperforms value
        - Size: Small caps outperform
    
    Attributes:
        factor_history: Historical factor returns
        
    Example:
        >>> analyzer = FactorExposureAnalyzer()
        >>> analyzer.update_regime(2, 0.9)  # Risk-Off Crisis
        >>> 
        >>> # Momentum crashing - expected in crisis
        >>> result = analyzer.analyze_factor("momentum", return_value=-0.05)
        >>> print(result["signal"].explanation)
        # "Momentum underperforming as expected in crisis"
    """
    
    STANDARD_FACTORS = ["momentum", "value", "quality", "size", "volatility", "growth"]
    
    # Fama-French 5 factor column names (as distributed by Ken French's library
    # and commonly reproduced in data vendors).  The mapping lets us bridge
    # between the academic column names and our internal generic names.
    FAMA_FRENCH_COLUMNS = {
        "Mkt-RF": "market",
        "SMB": "size",
        "HML": "value",
        "RMW": "quality",    # profitability ≈ quality
        "CMA": "growth",     # investment ≈ inverse growth
        "Mom": "momentum",   # Carhart momentum (separate file)
        "UMD": "momentum",   # alternative momentum label
        "RF": "_rf",          # risk-free rate (kept for alpha calc)
    }
    
    # AQR factor column aliases
    AQR_COLUMNS = {
        "MKT": "market",
        "SMB": "size",
        "HML_FF": "value",
        "UMD": "momentum",
        "BAB": "volatility",   # Betting-Against-Beta ~ low-vol
        "QMJ": "quality",      # Quality Minus Junk
    }
    
    # Expected factor performance by regime
    FACTOR_EXPECTATIONS = {
        1: {  # Risk-On Growth
            "momentum": {"expected": "positive", "weight": 1.2},
            "quality": {"expected": "positive", "weight": 1.0},
            "value": {"expected": "negative", "weight": 0.8},
            "size": {"expected": "positive", "weight": 1.1},  # small > large
            "volatility": {"expected": "negative", "weight": 0.9},  # low vol wins
            "growth": {"expected": "positive", "weight": 1.2},
        },
        2: {  # Risk-Off Crisis
            "momentum": {"expected": "negative", "weight": 1.0},
            "quality": {"expected": "positive", "weight": 1.3},
            "value": {"expected": "neutral", "weight": 0.8},
            "size": {"expected": "negative", "weight": 1.2},  # large > small
            "volatility": {"expected": "positive", "weight": 1.1},  # low vol wins
            "growth": {"expected": "negative", "weight": 0.9},
        },
        3: {  # Stagflation
            "momentum": {"expected": "negative", "weight": 0.9},
            "quality": {"expected": "neutral", "weight": 1.0},
            "value": {"expected": "positive", "weight": 1.1},  # real assets
            "size": {"expected": "negative", "weight": 0.9},
            "volatility": {"expected": "positive", "weight": 1.0},
            "growth": {"expected": "negative", "weight": 1.2},
        },
        4: {  # Disinflationary Boom
            "momentum": {"expected": "positive", "weight": 1.1},
            "quality": {"expected": "positive", "weight": 1.2},
            "value": {"expected": "negative", "weight": 0.8},
            "size": {"expected": "positive", "weight": 1.0},
            "volatility": {"expected": "neutral", "weight": 0.9},
            "growth": {"expected": "positive", "weight": 1.3},
        },
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the factor analyzer."""
        super().__init__(name="FactorExposureAnalyzer", config_path=config_path)
        
        self.factor_history: Dict[str, List[FactorReturn]] = {
            f: [] for f in self.STANDARD_FACTORS
        }
        self._historical_stats: Dict[str, Dict] = {}
        self._rolling_window: int = 60  # trading days for rolling OLS
        self._min_observations: int = 40
        self._latest_multi_ols: Optional[OLSResult] = None
    
    def get_regime_parameters(self, regime: int) -> Dict:
        """Get factor expectations for a specific regime."""
        return {
            "expectations": self.FACTOR_EXPECTATIONS.get(regime, {}),
        }
    
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze factor returns data using rolling OLS regression.
        
        If Fama-French or AQR columns are detected (e.g. ``Mkt-RF``,
        ``SMB``, ``HML``, ``RMW``, ``CMA``), a **multi-factor** rolling
        OLS is run that jointly estimates all factor loadings and alpha.
        Otherwise each factor is regressed individually (single-factor
        OLS) or analysed via z-score.
        
        Args:
            data: DataFrame with factor return columns and asset returns.
            
        Returns:
            Comprehensive factor analysis with OLS betas and rotations.
        """
        signals: List[ModuleSignal] = []
        factor_results: Dict[str, Any] = {}
        ols_betas: Dict[str, float] = {}
        multi_ols_result: Optional[OLSResult] = None

        # ----- Detect & normalise Fama-French / AQR columns ---------------
        mapped_data, ff_detected = self._normalise_factor_columns(data)

        # ----- Path A: Multi-factor OLS (Fama-French / AQR) ---------------
        dep_col = self._find_dependent_column(mapped_data)
        if ff_detected and dep_col is not None:
            multi_ols_result = self._run_multi_factor_ols(mapped_data, dep_col)
            if multi_ols_result is not None:
                self._latest_multi_ols = multi_ols_result
                ols_betas = dict(multi_ols_result.betas)
                logger.info(
                    "Multi-factor OLS: alpha=%.4f R²=%.3f betas=%s",
                    multi_ols_result.alpha, multi_ols_result.r_squared,
                    {k: round(v, 3) for k, v in multi_ols_result.betas.items()},
                )

        # ----- Analyse each factor ----------------------------------------
        analysis_data = mapped_data if ff_detected else data
        for factor in self.STANDARD_FACTORS:
            if factor in analysis_data.columns:
                returns = analysis_data[factor].dropna()
                if len(returns) == 0:
                    continue
                
                current_return = float(returns.iloc[-1])
                z_score = 0.0
                beta = ols_betas.get(factor)

                # Single-factor OLS fallback
                if beta is None and dep_col and len(returns) >= self._min_observations:
                    beta = self._compute_rolling_ols_beta(
                        analysis_data, dep_col, factor,
                    )
                    if beta is not None:
                        ols_betas[factor] = beta
                
                # Z-score via expanding window
                if len(returns) > 20:
                    mean = float(returns.iloc[:-1].mean())
                    std = float(returns.iloc[:-1].std())
                    z_score = (current_return - mean) / std if std > 0 else 0
                    self._historical_stats[factor] = {
                        "mean": mean,
                        "std": std,
                        "observations": len(returns) - 1,
                    }
                
                result = self.analyze_factor(
                    factor_name=factor,
                    return_value=current_return,
                    z_score=z_score,
                    beta=beta,
                )
                factor_results[factor] = result
                signals.append(result["signal"])
        
        # Factor rotation detection
        rotation = self.detect_factor_rotation()
        
        if signals:
            bullish = sum(1 for s in signals if s.signal == "bullish")
            bearish = sum(1 for s in signals if s.signal == "bearish")
            avg_strength = float(np.mean([s.strength for s in signals]))
            
            overall = "bullish" if bullish > bearish else "bearish" if bearish > bullish else "neutral"
            
            output: Dict[str, Any] = {
                "signal": self.create_signal(
                    signal=overall,
                    strength=avg_strength,
                    explanation=f"Factor analysis across {len(signals)} factors",
                ),
                "factor_results": factor_results,
                "ols_betas": ols_betas,
                "rotation": rotation,
                "historical_stats": self._historical_stats,
                "regime_parameters": self.get_current_parameters(),
            }
            if multi_ols_result is not None:
                output["multi_factor_ols"] = {
                    "alpha": multi_ols_result.alpha,
                    "r_squared": multi_ols_result.r_squared,
                    "t_stats": multi_ols_result.t_stats,
                    "residual_vol": multi_ols_result.residual_vol,
                    "n_obs": multi_ols_result.n_obs,
                }
            return output
        
        return {
            "signal": self.create_signal(
                signal="neutral",
                strength=0.0,
                explanation="No factor data available",
            ),
        }

    # ---- Column normalisation --------------------------------------------

    @staticmethod
    def _find_dependent_column(data: pd.DataFrame) -> Optional[str]:
        """Return the best dependent-variable column name, or *None*."""
        for candidate in ("SPX_returns", "SPX", "Mkt-RF", "MKT", "portfolio"):
            if candidate in data.columns:
                return candidate
        return None

    def _normalise_factor_columns(
        self, data: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, bool]:
        """Detect Fama-French / AQR columns and map to generic names.

        If both FF and AQR columns are present, the map with the most
        matching columns is preferred.

        Returns a (possibly augmented) copy of *data* and a boolean flag
        indicating whether academic-format columns were detected.

        Args:
            data: Raw input DataFrame.

        Returns:
            Tuple of (normalised DataFrame, True if FF/AQR columns found).
        """
        ff_cols = set(self.FAMA_FRENCH_COLUMNS) & set(data.columns)
        aqr_cols = set(self.AQR_COLUMNS) & set(data.columns)

        mapping: Dict[str, str] = {}

        # Prefer whichever map has more matching columns
        if len(aqr_cols) > len(ff_cols) and len(aqr_cols) >= 2:
            for col in aqr_cols:
                target = self.AQR_COLUMNS[col]
                if target not in data.columns:
                    mapping[col] = target
        elif len(ff_cols) >= 2:
            for col in ff_cols:
                target = self.FAMA_FRENCH_COLUMNS[col]
                if target != "_rf" and target not in data.columns:
                    mapping[col] = target
        elif len(aqr_cols) >= 2:
            for col in aqr_cols:
                target = self.AQR_COLUMNS[col]
                if target not in data.columns:
                    mapping[col] = target

        if not mapping:
            return data, False

        out = data.copy()
        out.rename(columns=mapping, inplace=True)
        logger.info("Normalised factor columns: %s", mapping)
        return out, True

    # ---- Multi-factor rolling OLS ----------------------------------------

    # Factors to consider for multi-factor OLS (STANDARD + market)
    _OLS_FACTORS = STANDARD_FACTORS + ["market"]

    def _run_multi_factor_ols(
        self,
        data: pd.DataFrame,
        dependent: str,
    ) -> Optional[OLSResult]:
        """Run multi-factor rolling OLS regression.

        Regresses *dependent* (e.g. portfolio or SPX excess returns)
        simultaneously on all available factor columns (including
        ``market`` when available from Fama-French / AQR mapping) over
        the trailing ``_rolling_window`` observations.

        Model: ``R_dep = alpha + sum(beta_i * factor_i) + epsilon``

        Args:
            data: DataFrame with dependent and factor columns.
            dependent: Name of the dependent variable column.

        Returns:
            :class:`OLSResult` with alpha, betas, R², t-stats,
            or *None* if insufficient data.
        """
        available = [f for f in self._OLS_FACTORS if f in data.columns and f != dependent]
        if not available or dependent not in data.columns:
            return None

        # Align & drop NAs
        cols = [dependent] + available
        aligned = data[cols].dropna()
        if len(aligned) < self._min_observations:
            return None

        window = aligned.iloc[-self._rolling_window:]
        y = window[dependent].values
        X_raw = window[available].values
        n, k = X_raw.shape

        # Prepend intercept
        X = np.column_stack([np.ones(n), X_raw])

        try:
            XtX = X.T @ X
            Xty = X.T @ y
            coeffs = np.linalg.solve(XtX, Xty)  # [alpha, beta1, …, betaK]
        except np.linalg.LinAlgError:
            return None

        alpha = float(coeffs[0])
        betas = {f: float(coeffs[i + 1]) for i, f in enumerate(available)}

        # Residuals & R²
        y_hat = X @ coeffs
        residuals = y - y_hat
        ss_res = float(np.sum(residuals ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        # Standard errors & t-stats
        t_stats: Dict[str, float] = {}
        residual_vol = 0.0
        if n > k + 1:
            mse = ss_res / (n - k - 1)
            residual_vol = float(np.sqrt(mse) * np.sqrt(252))  # annualised
            try:
                cov_matrix = mse * np.linalg.inv(XtX)
                se = np.sqrt(np.diag(cov_matrix))
                for i, f in enumerate(available):
                    t_stats[f] = float(coeffs[i + 1] / se[i + 1]) if se[i + 1] > 0 else 0.0
            except np.linalg.LinAlgError:
                pass

        return OLSResult(
            alpha=alpha,
            betas=betas,
            r_squared=r_squared,
            t_stats=t_stats,
            residual_vol=residual_vol,
            n_obs=n,
        )
    
    def _compute_rolling_ols_beta(
        self,
        data: pd.DataFrame,
        dependent: str,
        factor: str,
    ) -> Optional[float]:
        """Compute rolling OLS beta of factor exposure.
        
        Regresses asset returns (dependent) on factor returns (independent)
        using a rolling window to compute current factor beta/exposure.
        
        Args:
            data: DataFrame with both columns
            dependent: Dependent variable column name (e.g., SPX returns)
            factor: Independent variable column name (factor returns)
            
        Returns:
            Current rolling OLS beta, or None if insufficient data
        """
        if dependent not in data.columns or factor not in data.columns:
            return None
        
        # Get aligned data
        aligned = data[[dependent, factor]].dropna()
        
        if len(aligned) < self._min_observations:
            return None
        
        # Use last rolling_window observations
        window_data = aligned.iloc[-self._rolling_window:]
        
        y = window_data[dependent].values
        x = window_data[factor].values
        
        # Add constant for OLS: y = alpha + beta * x
        X = np.column_stack([np.ones(len(x)), x])
        
        try:
            # OLS via normal equation: (X'X)^-1 X'y
            XtX = X.T @ X
            Xty = X.T @ y
            betas = np.linalg.solve(XtX, Xty)
            return float(betas[1])  # Return the slope (beta)
        except np.linalg.LinAlgError:
            return None
    
    def analyze_factor(
        self,
        factor_name: str,
        return_value: float,
        z_score: Optional[float] = None,
        percentile: Optional[float] = None,
        beta: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Analyze a single factor's performance.
        
        Args:
            factor_name: Name of the factor
            return_value: Current period return
            z_score: Z-score of return
            percentile: Historical percentile
            beta: Rolling OLS beta (factor exposure coefficient)
            
        Returns:
            Factor analysis with regime-adaptive interpretation
        """
        expectations = self.FACTOR_EXPECTATIONS.get(
            self.current_regime, {}
        ).get(factor_name.lower(), {"expected": "neutral", "weight": 1.0})
        
        expected = expectations["expected"]
        weight = expectations["weight"]
        regime_name = self.REGIME_NAMES[self.current_regime]
        
        # Determine if factor is behaving as expected
        if expected == "positive":
            if return_value > 0.01:  # 1% threshold
                behaving_as_expected = True
                signal = "bullish"
                explanation = f"{factor_name} positive as expected in {regime_name}"
            elif return_value < -0.01:
                behaving_as_expected = False
                signal = "cautious"
                explanation = f"{factor_name} negative (unexpected in {regime_name})"
            else:
                behaving_as_expected = True
                signal = "neutral"
                explanation = f"{factor_name} flat in {regime_name}"
        elif expected == "negative":
            if return_value < -0.01:
                behaving_as_expected = True
                signal = "neutral"  # Expected underperformance is not a signal
                explanation = f"{factor_name} negative as expected in {regime_name}"
            elif return_value > 0.01:
                behaving_as_expected = False
                signal = "bullish"
                explanation = f"{factor_name} positive (unexpected in {regime_name})"
            else:
                behaving_as_expected = True
                signal = "neutral"
                explanation = f"{factor_name} flat in {regime_name}"
        else:  # neutral expectation
            signal = "neutral"
            behaving_as_expected = True
            explanation = f"{factor_name} neutral in {regime_name}"
        
        # Strength based on z-score if available
        if z_score is not None:
            strength = min(1.0, abs(z_score) / 3)
        else:
            strength = min(1.0, abs(return_value) * 10)
        
        # Weight by regime importance
        strength *= weight
        strength = min(1.0, strength)
        
        # Store observation
        factor_obs = FactorReturn(
            factor=factor_name,
            return_value=return_value,
            z_score=z_score or 0,
            percentile=percentile or 50,
        )
        if factor_name.lower() in self.factor_history:
            self.factor_history[factor_name.lower()].append(factor_obs)
        
        return {
            "signal": self.create_signal(
                signal=signal,
                strength=strength,
                explanation=explanation,
                factor=factor_name,
                return_value=return_value,
            ),
            "expected": expected,
            "behaving_as_expected": behaving_as_expected,
            "z_score": z_score,
            "beta": beta,
        }
    
    def detect_factor_rotation(
        self,
        lookback: int = 20,
    ) -> Dict[str, Any]:
        """Detect factor rotation patterns.
        
        Args:
            lookback: Periods to analyze
            
        Returns:
            Factor rotation signals
        """
        rotations = {}
        
        for factor, history in self.factor_history.items():
            if len(history) >= lookback:
                recent = history[-lookback:]
                first_half = recent[:lookback//2]
                second_half = recent[lookback//2:]
                
                avg_first = np.mean([r.return_value for r in first_half])
                avg_second = np.mean([r.return_value for r in second_half])
                
                change = avg_second - avg_first
                
                if change > 0.02:
                    rotations[factor] = "improving"
                elif change < -0.02:
                    rotations[factor] = "deteriorating"
                else:
                    rotations[factor] = "stable"
        
        return {
            "rotations": rotations,
            "interpretation": self._interpret_rotations(rotations),
        }
    
    def _interpret_rotations(self, rotations: Dict[str, str]) -> str:
        """Interpret factor rotation pattern."""
        improving = [f for f, r in rotations.items() if r == "improving"]
        deteriorating = [f for f, r in rotations.items() if r == "deteriorating"]
        
        if "quality" in improving and "momentum" in deteriorating:
            return "Rotation to defensives - possible regime shift to risk-off"
        elif "momentum" in improving and "value" in deteriorating:
            return "Growth momentum continuing - consistent with risk-on"
        elif "value" in improving and "growth" in deteriorating:
            return "Value rotation - possible regime shift"
        else:
            return "No clear rotation pattern"
