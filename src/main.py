"""
AMRCAIS - Adaptive Multi-Regime Cross-Asset Intelligence System.

Main entry point for the regime detection and analysis system.

Usage:
    python -m src.main --mode=analyze --symbols SPX,TLT,GLD
    python -m src.main --mode=backtest --start 2020-01-01 --end 2024-01-01
"""

import argparse
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from src.data_pipeline.pipeline import DataPipeline
from src.regime_detection.ensemble import RegimeEnsemble
from src.modules import (
    MacroEventTracker,
    YieldCurveAnalyzer,
    OptionsSurfaceMonitor,
    FactorExposureAnalyzer,
    CorrelationAnomalyDetector,
)
from src.meta_learning import MetaLearner

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('amrcais.log')
    ]
)
logger = logging.getLogger(__name__)


class AMRCAIS:
    """Main AMRCAIS system coordinator.
    
    Coordinates the three-layer architecture:
    - Layer 1: Regime Detection (4 classifiers + ensemble)
    - Layer 2: Analytical Modules (5 modules with regime-adaptive signals)
    - Layer 3: Meta-Learning (performance tracking, recalibration)
    
    Example:
        >>> system = AMRCAIS()
        >>> system.initialize()
        >>> result = system.analyze()
        >>> print(f"Current Regime: {result['regime']['name']}")
        >>> print(f"Disagreement: {result['regime']['disagreement']:.2f}")
    """
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        db_path: Optional[str] = None,
    ):
        """Initialize AMRCAIS.
        
        Args:
            config_path: Path to configuration directory
            db_path: Path to SQLite database
        """
        self.config_path = config_path or "config"
        self.db_path = db_path or "data/amrcais.db"
        
        # Core components
        self.pipeline: Optional[DataPipeline] = None
        self.ensemble: Optional[RegimeEnsemble] = None
        
        # Meta-learning layer (Phase 3)
        self.meta_learner: Optional[MetaLearner] = None
        
        # Analytical modules
        self.modules: Dict = {}
        
        # State
        self._is_initialized = False
        self._current_regime: Optional[int] = None
        self._current_confidence: float = 0.0
        self._current_disagreement: float = 0.0
        
        logger.info("AMRCAIS instance created")
    
    def initialize(self, lookback_days: int = 365) -> None:
        """Initialize system with historical data.
        
        Args:
            lookback_days: Days of history to load for training
        """
        logger.info(f"Initializing AMRCAIS with {lookback_days} days lookback")
        
        # Initialize data pipeline
        self.pipeline = DataPipeline(db_path=self.db_path)
        
        # Fetch initial data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        try:
            self.market_data = self.pipeline.fetch_market_data(
                start_date=start_date,
                end_date=end_date,
            )
            logger.info(f"Loaded market data: {self.market_data.shape}")
        except Exception as e:
            logger.warning(f"Could not fetch live data: {e}. Using cached if available.")
            self.market_data = pd.DataFrame()
        
        # Initialize ensemble
        self.ensemble = RegimeEnsemble(config_path=self.config_path)
        
        if len(self.market_data) > 100:
            self.ensemble.fit(self.market_data)
            logger.info("Ensemble classifier fitted")
        
        # Initialize analytical modules
        self.modules = {
            "macro": MacroEventTracker(config_path=self.config_path),
            "yield_curve": YieldCurveAnalyzer(config_path=self.config_path),
            "options": OptionsSurfaceMonitor(config_path=self.config_path),
            "factors": FactorExposureAnalyzer(config_path=self.config_path),
            "correlations": CorrelationAnomalyDetector(config_path=self.config_path),
        }
        
        # Initialize meta-learning layer (Phase 3 - The Killer Feature)
        storage_path = Path(self.db_path).parent / "regime_history.csv"
        self.meta_learner = MetaLearner(storage_path=storage_path)
        logger.info("Meta-learning layer initialized")
        
        self._is_initialized = True
        logger.info("AMRCAIS initialization complete with adaptive learning enabled")
    
    def analyze(self, data: Optional[pd.DataFrame] = None) -> Dict:
        """Run full analysis.
        
        Args:
            data: Optional new data to analyze (uses latest if not provided)
            
        Returns:
            Complete analysis results with regime and module signals
        """
        if not self._is_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        if data is None:
            data = self.market_data
        
        if len(data) == 0:
            logger.warning("No data available for analysis")
            return {"error": "No data available"}
        
        results = {}
        
        # Layer 1: Get regime
        regime_result = self.ensemble.predict(data)
        self._current_regime = regime_result.regime
        self._current_confidence = regime_result.confidence
        self._current_disagreement = regime_result.disagreement
        
        results["regime"] = {
            "id": regime_result.regime,
            "name": regime_result.regime_name,
            "confidence": regime_result.confidence,
            "disagreement": regime_result.disagreement,
            "transition_warning": regime_result.transition_warning,
            "probabilities": regime_result.probabilities,
            "individual_predictions": regime_result.individual_predictions,
        }
        
        # Log to meta-learner for adaptive tracking
        if self.meta_learner:
            # Extract market state for validation
            market_state = {}
            if len(data) > 0:
                latest = data.iloc[-1] if hasattr(data, 'iloc') else {}
                if isinstance(latest, pd.Series):
                    market_state = latest.to_dict()
            
            self.meta_learner.log_classification(
                regime=regime_result.regime,
                confidence=regime_result.confidence,
                disagreement=regime_result.disagreement,
                individual_predictions=regime_result.individual_predictions,
                market_state=market_state,
            )
        
        # Layer 2: Update modules with regime and get signals
        module_signals = {}
        
        for name, module in self.modules.items():
            # Update module with current regime
            module.update_regime(regime_result.regime, regime_result.confidence)
            
            # Get module analysis
            try:
                module_result = module.analyze(data)
                module_signals[name] = {
                    "signal": module_result.get("signal", {}).to_dict() 
                             if hasattr(module_result.get("signal", {}), "to_dict") 
                             else str(module_result.get("signal", "neutral")),
                    "details": {k: v for k, v in module_result.items() if k != "signal"},
                }
            except Exception as e:
                logger.warning(f"Module {name} analysis failed: {e}")
                module_signals[name] = {"error": str(e)}
        
        results["modules"] = module_signals
        
        # Layer 3: Meta-learning checks
        if self.meta_learner:
            recalibration_decision = self.meta_learner.check_recalibration_needed()
            uncertainty_signal = self.meta_learner.generate_uncertainty_signal()
            adaptive_weights = self.meta_learner.get_adaptive_weights()
            
            results["meta"] = {
                "needs_recalibration": recalibration_decision.should_recalibrate,
                "recalibration_urgency": recalibration_decision.urgency_level,
                "recalibration_severity": recalibration_decision.severity,
                "recalibration_reasons": [str(r) for r in recalibration_decision.reasons],
                "uncertainty_signal": uncertainty_signal,
                "adaptive_weights": adaptive_weights,
                "performance_metrics": self.meta_learner.get_performance_metrics(30).to_dict(),
            }
        else:
            # Fallback if meta-learner not initialized
            needs_recal, recal_reason = self.ensemble.needs_recalibration()
            results["meta"] = {
                "needs_recalibration": needs_recal,
                "recalibration_reason": recal_reason,
                "classifier_performance": self.ensemble.get_classifier_performance(),
            }
        
        # Generate summary
        results["summary"] = self._generate_summary(results)
        
        return results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate human-readable summary."""
        regime = results["regime"]
        
        summary = {
            "headline": f"Current Regime: {regime['name']}",
            "confidence_level": "High" if regime["confidence"] > 0.7 else 
                               "Medium" if regime["confidence"] > 0.5 else "Low",
            "stability": "Unstable" if regime["transition_warning"] else "Stable",
        }
        
        # Count bullish/bearish signals
        bullish = 0
        bearish = 0
        
        for name, module in results["modules"].items():
            signal = module.get("signal", {})
            if isinstance(signal, dict):
                if signal.get("signal") == "bullish":
                    bullish += 1
                elif signal.get("signal") == "bearish":
                    bearish += 1
        
        if bullish > bearish:
            summary["overall_bias"] = "Bullish"
        elif bearish > bullish:
            summary["overall_bias"] = "Bearish"
        else:
            summary["overall_bias"] = "Neutral"
        
        summary["signal_counts"] = {"bullish": bullish, "bearish": bearish}
        
        return summary
    
    def get_current_state(self) -> Dict:
        """Get current system state."""
        return {
            "regime": self._current_regime,
            "confidence": self._current_confidence,
            "disagreement": self._current_disagreement,
            "is_initialized": self._is_initialized,
        }
    
    def recalibrate(self, new_data: Optional[pd.DataFrame] = None) -> None:
        """Recalibrate the system."""
        if not self._is_initialized:
            raise RuntimeError("System not initialized")
        
        data = new_data if new_data is not None else self.market_data
        
        logger.info("Recalibrating AMRCAIS...")
        self.ensemble.recalibrate(data)
        logger.info("Recalibration complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AMRCAIS - Market Regime Analysis")
    parser.add_argument("--mode", choices=["analyze", "backtest", "monitor"],
                       default="analyze", help="Operation mode")
    parser.add_argument("--lookback", type=int, default=365,
                       help="Days of historical data")
    parser.add_argument("--config", type=str, default="config",
                       help="Configuration directory")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("AMRCAIS - Adaptive Multi-Regime Cross-Asset Intelligence System")
    print("=" * 60)
    
    # Initialize system
    system = AMRCAIS(config_path=args.config)
    
    try:
        system.initialize(lookback_days=args.lookback)
    except Exception as e:
        logger.error(f"Initialization failed: {e}")
        print(f"\nNote: Some data sources may require API keys.")
        print("Set FRED_API_KEY and/or ALPHA_VANTAGE_API_KEY environment variables.")
        return 1
    
    if args.mode == "analyze":
        # Run analysis
        results = system.analyze()
        
        # Print results
        print("\n" + "=" * 40)
        print("REGIME ANALYSIS")
        print("=" * 40)
        
        regime = results.get("regime", {})
        print(f"\nCurrent Regime: {regime.get('name', 'Unknown')}")
        print(f"Confidence: {regime.get('confidence', 0):.1%}")
        print(f"Disagreement Index: {regime.get('disagreement', 0):.2f}")
        
        if regime.get("transition_warning"):
            print("\n⚠️  WARNING: High disagreement suggests possible regime transition")
        
        print("\nRegime Probabilities:")
        for r, p in regime.get("probabilities", {}).items():
            print(f"  Regime {r}: {p:.1%}")
        
        print("\n" + "=" * 40)
        print("MODULE SIGNALS")
        print("=" * 40)
        
        for name, module in results.get("modules", {}).items():
            signal = module.get("signal", {})
            if isinstance(signal, dict):
                print(f"\n{name.upper()}:")
                print(f"  Signal: {signal.get('signal', 'N/A')}")
                print(f"  Strength: {signal.get('strength', 0):.2f}")
        
        summary = results.get("summary", {})
        print("\n" + "=" * 40)
        print("SUMMARY")
        print("=" * 40)
        print(f"\n{summary.get('headline', 'No summary')}")
        print(f"Confidence: {summary.get('confidence_level', 'N/A')}")
        print(f"Stability: {summary.get('stability', 'N/A')}")
        print(f"Overall Bias: {summary.get('overall_bias', 'N/A')}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
