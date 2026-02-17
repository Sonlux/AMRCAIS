"""
Base Analytical Module for AMRCAIS.

This module defines the abstract interface that all analytical modules
must implement. The key design principle is REGIME-FIRST: every module
must adapt its parameters and signal interpretation based on the current
market regime.

Classes:
    AnalyticalModule: Abstract base class for all analytical modules
    ModuleSignal: Dataclass containing module output signal

Example:
    >>> class MyModule(AnalyticalModule):
    ...     def analyze(self, data):
    ...         regime = self.current_regime
    ...         params = self.get_regime_parameters(regime)
    ...         # Regime-adaptive analysis
    ...         return {"signal": "bullish", "strength": 0.8}
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

import pandas as pd
import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModuleSignal:
    """Output signal from an analytical module.
    
    Attributes:
        signal: Signal direction ("bullish", "bearish", "neutral")
        strength: Signal strength (0-1)
        explanation: Human-readable explanation of the signal
        regime_context: How the regime affected interpretation
        confidence: Confidence in the signal (0-1)
        timestamp: When the signal was generated
        metadata: Additional module-specific data
    """
    signal: str  # "bullish", "bearish", "neutral", "cautious"
    strength: float  # 0 to 1
    explanation: str
    regime_context: str
    confidence: float = 0.5
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate signal fields."""
        valid_signals = {"bullish", "bearish", "neutral", "cautious"}
        if self.signal not in valid_signals:
            raise ValueError(f"Signal must be one of {valid_signals}, got {self.signal}")
        
        if not 0 <= self.strength <= 1:
            raise ValueError(f"Strength must be 0-1, got {self.strength}")
        
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be 0-1, got {self.confidence}")
        
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def direction(self) -> int:
        """Numeric signal direction: 1 for bullish, -1 for bearish, 0 for neutral/cautious."""
        return {"bullish": 1, "bearish": -1, "neutral": 0, "cautious": 0}[self.signal]
    
    @property
    def weighted_signal(self) -> float:
        """Signal strength weighted by direction and confidence."""
        return self.direction * self.strength * self.confidence
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "signal": self.signal,
            "strength": self.strength,
            "explanation": self.explanation,
            "regime_context": self.regime_context,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "weighted_signal": self.weighted_signal,
            "metadata": self.metadata,
        }


class AnalyticalModule(ABC):
    """Abstract base class for all AMRCAIS analytical modules.
    
    Defines the interface that all modules (MacroEventTracker, YieldCurveAnalyzer,
    OptionsSurfaceMonitor, FactorExposureAnalyzer, CorrelationAnomalyDetector)
    must implement.
    
    CRITICAL DESIGN PRINCIPLE: REGIME-FIRST
    Every module must adapt its analysis based on the current market regime.
    The same data can produce different signals in different regimes.
    
    Example:
        >>> module = YieldCurveAnalyzer()
        >>> module.update_regime(regime=1, confidence=0.8)  # Risk-On Growth
        >>> result = module.analyze(yield_curve_data)
        >>> print(result["signal"])  # Regime-adapted interpretation
    
    Attributes:
        current_regime: Current market regime (1-4)
        regime_confidence: Confidence in current regime classification
        config: Module configuration from YAML
        name: Module name for logging
    """
    
    # Regime definitions
    REGIME_NAMES = {
        1: "Risk-On Growth",
        2: "Risk-Off Crisis",
        3: "Stagflation",
        4: "Disinflationary Boom",
    }
    
    def __init__(
        self,
        name: str,
        config_path: Optional[str] = None,
    ):
        """Initialize the analytical module.
        
        Args:
            name: Module name for logging and identification
            config_path: Path to configuration YAML file
        """
        self.name = name
        self.current_regime: int = 1
        self.regime_confidence: float = 0.5
        self._regime_params: Dict[int, Dict] = {}
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize regime-specific parameters
        self._load_regime_parameters()
        
        logger.info(f"Initialized {self.name} module")
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load module configuration from YAML."""
        if config_path:
            p = Path(config_path)
            # If a directory was passed, look for regimes.yaml inside it
            if p.is_dir():
                regimes_file = p / "regimes.yaml"
                if regimes_file.is_file():
                    with open(regimes_file, "r") as f:
                        config = yaml.safe_load(f) or {}
                        logger.debug(f"Loaded config from {regimes_file}")
                        return config
            elif p.is_file():
                with open(p, "r") as f:
                    return yaml.safe_load(f) or {}
        
        # Try default paths
        default_paths = [
            "config/regimes.yaml",
            "../config/regimes.yaml",
            Path(__file__).parent.parent.parent / "config" / "regimes.yaml",
        ]
        
        for path in default_paths:
            path = Path(path)
            if path.is_file():
                with open(path, "r") as f:
                    config = yaml.safe_load(f) or {}
                    logger.debug(f"Loaded config from {path}")
                    return config
        
        logger.warning(f"No config file found for {self.name}")
        return {}
    
    def _load_regime_parameters(self) -> None:
        """Load regime-specific parameters from config."""
        regimes_config = self.config.get("regimes", {})
        
        for regime_id in range(1, 5):
            regime_config = regimes_config.get(regime_id, {})
            self._regime_params[regime_id] = regime_config
    
    def update_regime(self, regime: int, confidence: float) -> None:
        """Update the current regime context.
        
        This method MUST be called when the regime changes to ensure
        the module adapts its analysis parameters.
        
        Args:
            regime: New regime ID (1-4)
            confidence: Confidence in the regime classification (0-1)
            
        Raises:
            ValueError: If regime is not 1-4 or confidence not 0-1
        """
        if not 1 <= regime <= 4:
            raise ValueError(f"Regime must be 1-4, got {regime}")
        if not 0 <= confidence <= 1:
            raise ValueError(f"Confidence must be 0-1, got {confidence}")
        
        if regime != self.current_regime:
            logger.info(
                f"{self.name}: Regime changed {self.current_regime} → {regime} "
                f"({self.REGIME_NAMES[regime]}), confidence={confidence:.2f}"
            )
        
        self.current_regime = regime
        self.regime_confidence = confidence
        
        # Trigger module-specific regime change handling
        self._on_regime_change(regime, confidence)
    
    def _on_regime_change(self, regime: int, confidence: float) -> None:
        """Hook for subclasses to react to regime changes.
        
        Override this method to perform module-specific actions
        when the regime changes (e.g., recalculate thresholds).
        """
        pass
    
    @abstractmethod
    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Perform regime-adaptive analysis on input data.
        
        This is the main entry point for module analysis. The implementation
        MUST consider self.current_regime when interpreting results.
        
        Args:
            data: Input data for analysis (format depends on module)
            
        Returns:
            Dictionary containing:
                - signal: ModuleSignal with direction and strength
                - raw_metrics: Raw calculated metrics
                - regime_context: How regime affected interpretation
                - explanation: Human-readable explanation
        """
        pass
    
    @abstractmethod
    def get_regime_parameters(self, regime: int) -> Dict:
        """Get module parameters specific to a regime.
        
        Each module must define what parameters change by regime.
        
        Args:
            regime: Regime ID (1-4)
            
        Returns:
            Dictionary of regime-specific parameters
        """
        pass
    
    def get_current_parameters(self) -> Dict:
        """Get parameters for the current regime."""
        return self.get_regime_parameters(self.current_regime)
    
    def create_signal(
        self,
        signal: str,
        strength: float,
        explanation: str,
        **metadata,
    ) -> ModuleSignal:
        """Helper to create a ModuleSignal with regime context.
        
        Args:
            signal: Signal direction ("bullish", "bearish", "neutral")
            strength: Signal strength (0-1)
            explanation: Human-readable explanation
            **metadata: Additional metadata to include
            
        Returns:
            ModuleSignal with current regime context
        """
        regime_name = self.REGIME_NAMES[self.current_regime]
        regime_context = (
            f"Signal generated in {regime_name} regime "
            f"(confidence={self.regime_confidence:.2f})"
        )
        
        return ModuleSignal(
            signal=signal,
            strength=strength,
            explanation=explanation,
            regime_context=regime_context,
            confidence=self.regime_confidence,
            timestamp=datetime.now(),
            metadata=metadata,
        )
    
    def interpret_with_regime(
        self,
        metric_name: str,
        metric_value: float,
        interpretations: Dict[int, Dict[str, str]],
        threshold: float = 0,
    ) -> Tuple[str, str]:
        """Interpret a metric value based on current regime.
        
        CRITICAL: This method embodies the core AMRCAIS principle that
        the same metric can have different meanings in different regimes.
        
        Args:
            metric_name: Name of the metric being interpreted
            metric_value: Current value of the metric
            interpretations: Dict mapping regime -> {positive: signal, negative: signal}
            threshold: Value at which metric changes interpretation
            
        Returns:
            Tuple of (signal, explanation)
            
        Example:
            >>> # Yield curve steepening interpretation varies by regime
            >>> interp = {
            ...     1: {"positive": "bullish", "negative": "bearish"},  # Risk-On
            ...     2: {"positive": "flight_to_quality", "negative": "recession"},
            ...     3: {"positive": "bearish", "negative": "disinflation_hope"},
            ...     4: {"positive": "neutral", "negative": "goldilocks"},
            ... }
            >>> signal, expl = self.interpret_with_regime(
            ...     "yield_curve_slope", 0.5, interp
            ... )
        """
        regime_interp = interpretations.get(self.current_regime, {})
        
        if metric_value > threshold:
            signal = regime_interp.get("positive", "neutral")
        elif metric_value < -threshold:
            signal = regime_interp.get("negative", "neutral")
        else:
            signal = "neutral"
        
        # Map special interpretations to standard signals
        signal_map = {
            "flight_to_quality": "neutral",
            "recession": "bearish",
            "disinflation_hope": "bullish",
            "goldilocks": "bullish",
        }
        final_signal = signal_map.get(signal, signal)
        
        explanation = (
            f"{metric_name} = {metric_value:.3f}. "
            f"In {self.REGIME_NAMES[self.current_regime]}, this is interpreted as '{signal}'"
        )
        
        return final_signal, explanation
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"{self.name}(regime={self.current_regime}, "
            f"confidence={self.regime_confidence:.2f})"
        )
