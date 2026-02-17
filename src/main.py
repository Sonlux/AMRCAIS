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
from src.modules.contagion_network import ContagionNetwork
from src.modules.macro_surprise_decay import SurpriseDecayModel
from src.regime_detection.transition_model import RegimeTransitionModel
from src.regime_detection.multi_timeframe import MultiTimeframeDetector
from src.narrative.narrative_generator import NarrativeGenerator
from src.meta_learning import MetaLearner
from src.prediction.return_forecaster import ReturnForecaster
from src.prediction.tail_risk import TailRiskAnalyzer
from src.prediction.portfolio_optimizer import PortfolioOptimizer
from src.prediction.alpha_signals import AlphaSignalGenerator
from src.regime_detection.ml_classifier import (
    create_training_labels,
    KNOWN_REGIME_PERIODS,
)

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
        
        # Phase 2 components
        self.transition_model: Optional[RegimeTransitionModel] = None
        self.multi_timeframe: Optional[MultiTimeframeDetector] = None
        self.narrative_generator: Optional[NarrativeGenerator] = None
        
        # Phase 3 components (Prediction Engine)
        self.return_forecaster: Optional[ReturnForecaster] = None
        self.tail_risk: Optional[TailRiskAnalyzer] = None
        self.portfolio_optimizer: Optional[PortfolioOptimizer] = None
        self.alpha_signals: Optional[AlphaSignalGenerator] = None
        
        # Phase 4 components (Real-Time + Execution)
        self.event_bus: Optional['EventBus'] = None
        self.scheduler: Optional['AnalysisScheduler'] = None
        self.alert_engine: Optional['AlertEngine'] = None
        self.stream_manager: Optional['StreamManager'] = None
        self.paper_trading: Optional['PaperTradingEngine'] = None
        
        # Phase 5 components (Network Effects + Moat)
        self.knowledge_base: Optional['KnowledgeBase'] = None
        self.alt_data_integrator: Optional['AltDataIntegrator'] = None
        self.research_publisher: Optional['ResearchPublisher'] = None
        self.user_manager: Optional['UserManager'] = None
        
        logger.info("AMRCAIS instance created")
    
    def initialize(self, lookback_days: int = 730) -> None:
        """Initialize system with historical data.
        
        Args:
            lookback_days: Calendar days of history to load (default 730 = ~2 years).
                The HMM classifier needs >=252 trading-day returns, and the
                ML classifier benefits from overlap with known regime periods
                (2020-2024), so 2 years is the practical minimum.
        """
        logger.info(f"Initializing AMRCAIS with {lookback_days} days lookback")
        
        # Initialize data pipeline
        self.pipeline = DataPipeline(db_path=self.db_path)
        
        # Fetch initial data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        try:
            raw_data = self.pipeline.fetch_market_data(
                start_date=start_date,
                end_date=end_date,
            )
            # fetch_market_data returns Dict[str, DataFrame]; merge Close
            # prices into a single DataFrame with one column per asset.
            if isinstance(raw_data, dict) and raw_data:
                close_frames = {}
                for asset, df in raw_data.items():
                    if isinstance(df, pd.DataFrame) and "Close" in df.columns:
                        close_frames[asset] = df["Close"]
                if close_frames:
                    self.market_data = pd.DataFrame(close_frames).dropna()
                else:
                    self.market_data = pd.DataFrame()
            elif isinstance(raw_data, pd.DataFrame):
                self.market_data = raw_data
            else:
                self.market_data = pd.DataFrame()
            logger.info(f"Loaded market data: {self.market_data.shape}")
        except Exception as e:
            logger.warning(f"Could not fetch live data: {e}. Using cached if available.")
            self.market_data = pd.DataFrame()
        
        # Initialize ensemble
        self.ensemble = RegimeEnsemble(config_path=self.config_path)
        
        if len(self.market_data) > 100:
            # Generate training labels for the ML classifier from known periods
            labels = None
            if hasattr(self.market_data, 'index'):
                labels = create_training_labels(
                    self.market_data.index, KNOWN_REGIME_PERIODS
                )
                labeled_count = int((labels != 0).sum())
                if labeled_count < 100:
                    logger.info(
                        f"Only {labeled_count} labeled days — ML classifier "
                        f"will skip (need >=100)"
                    )
                    labels = None
                else:
                    logger.info(f"Generated {labeled_count} training labels for ML classifier")
            
            self.ensemble.fit(self.market_data, labels=labels)
            logger.info("Ensemble classifier fitted")
        
        # Initialize analytical modules
        self.modules = {
            "macro": MacroEventTracker(config_path=self.config_path),
            "yield_curve": YieldCurveAnalyzer(config_path=self.config_path),
            "options": OptionsSurfaceMonitor(config_path=self.config_path),
            "factors": FactorExposureAnalyzer(config_path=self.config_path),
            "correlations": CorrelationAnomalyDetector(config_path=self.config_path),
            "contagion": ContagionNetwork(config_path=self.config_path),
            "surprise_decay": SurpriseDecayModel(config_path=self.config_path),
        }
        
        # Initialize Phase 2 components
        self.transition_model = RegimeTransitionModel()
        self.narrative_generator = NarrativeGenerator()
        
        # Multi-timeframe detector (fit if enough data)
        self.multi_timeframe = MultiTimeframeDetector(config_path=self.config_path)
        if len(self.market_data) >= 252:
            try:
                self.multi_timeframe.fit(self.market_data)
                logger.info("Multi-timeframe detector fitted")
            except Exception as e:
                logger.warning(f"Multi-timeframe fitting failed: {e}")
        
        # Fit transition model if ensemble is fitted and enough data
        if self.ensemble and self.ensemble.is_fitted and len(self.market_data) > 100:
            try:
                from src.regime_detection.ensemble import EnsembleResult
                regime_series = pd.Series(
                    [r.regime for r in self.ensemble.predict_sequence(
                        self.market_data, window=60, step=1
                    )],
                    index=self.market_data.index[60:],
                )
                self.transition_model.fit(self.market_data, regime_series)
                logger.info("Transition probability model fitted")
            except Exception as e:
                logger.warning(f"Transition model fitting failed: {e}")
        
        # Initialize meta-learning layer (Phase 3 - The Killer Feature)
        storage_path = Path(self.db_path).parent / "regime_history.csv"
        self.meta_learner = MetaLearner(storage_path=storage_path)
        logger.info("Meta-learning layer initialized")
        
        # Initialize Phase 3: Prediction Engine
        self.return_forecaster = ReturnForecaster()
        self.tail_risk = TailRiskAnalyzer()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.alpha_signals = AlphaSignalGenerator()
        
        # Fit Phase 3 components if enough data and ensemble fitted
        if self.ensemble and self.ensemble.is_fitted and len(self.market_data) > 100:
            try:
                from src.regime_detection.ensemble import EnsembleResult
                # Build regime series for prediction engine
                if not hasattr(self, '_regime_series') or self._regime_series is None:
                    predictions = self.ensemble.predict_sequence(
                        self.market_data, window=60, step=1
                    )
                    self._regime_series = pd.Series(
                        [r.regime for r in predictions],
                        index=self.market_data.index[60:],
                    )
                
                self.return_forecaster.fit(self.market_data, self._regime_series)
                self.tail_risk.fit(self.market_data, self._regime_series)
                self.portfolio_optimizer.fit(self.market_data, self._regime_series)
                self.alpha_signals.fit(self.market_data, self._regime_series)
                logger.info("Phase 3 Prediction Engine fitted")
            except Exception as e:
                logger.warning(f"Phase 3 fitting failed: {e}")
        
        # Initialize Phase 4: Real-Time + Execution
        try:
            from src.realtime.event_bus import EventBus
            from src.realtime.alert_engine import AlertEngine
            from src.realtime.stream_manager import StreamManager
            from src.realtime.paper_trading import PaperTradingEngine

            self.event_bus = EventBus()
            self.alert_engine = AlertEngine(self.event_bus)
            self.alert_engine.start()
            self.stream_manager = StreamManager(self.event_bus)
            self.stream_manager.start()
            self.paper_trading = PaperTradingEngine(
                self.event_bus, initial_capital=100_000.0
            )
            logger.info("Phase 4 Real-Time components initialized")
        except Exception as e:
            logger.warning(f"Phase 4 initialization failed: {e}")

        # Initialize Phase 5: Network Effects + Moat
        try:
            from src.knowledge.knowledge_base import KnowledgeBase
            from src.knowledge.alt_data import AltDataIntegrator
            from src.knowledge.research_publisher import ResearchPublisher
            from src.knowledge.user_manager import UserManager

            kb_path = str(Path(self.db_path).parent / "knowledge.json")
            users_path = str(Path(self.db_path).parent / "users.json")
            reports_dir = str(Path(self.db_path).parent / "reports")

            self.knowledge_base = KnowledgeBase(storage_path=kb_path)
            self.alt_data_integrator = AltDataIntegrator()
            self.research_publisher = ResearchPublisher(
                knowledge_base=self.knowledge_base,
                output_dir=reports_dir,
            )
            self.user_manager = UserManager(storage_path=users_path)
            logger.info("Phase 5 Network Effects components initialized")
        except Exception as e:
            logger.warning(f"Phase 5 initialization failed: {e}")

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
        
        # Phase 5: Feed regime transitions into the knowledge base
        if self.knowledge_base:
            prev_regime = getattr(self, "_prev_regime", None)
            if prev_regime is not None and prev_regime != regime_result.regime:
                try:
                    self.knowledge_base.record_transition(
                        from_regime=prev_regime,
                        to_regime=regime_result.regime,
                        confidence=regime_result.confidence,
                        disagreement=regime_result.disagreement,
                        leading_indicators={
                            "confidence": regime_result.confidence,
                            "disagreement": regime_result.disagreement,
                        },
                        classifier_accuracy={
                            clf: (pred == regime_result.regime)
                            for clf, pred in regime_result.individual_predictions.items()
                        },
                    )
                except Exception as kb_err:
                    logger.debug(f"Knowledge base transition recording skipped: {kb_err}")
            self._prev_regime = regime_result.regime
        
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
                signal_obj = module_result.get("signal", {})
                signal_dict = (
                    signal_obj.to_dict()
                    if hasattr(signal_obj, "to_dict")
                    else {"signal": str(signal_obj)} if signal_obj else {"signal": "neutral"}
                )
                module_signals[name] = {
                    "signal": signal_dict,
                    "details": {k: v for k, v in module_result.items() if k != "signal"},
                }
                
                # Persist signal to database for history tracking
                if self.pipeline and hasattr(self.pipeline, "storage"):
                    try:
                        import json
                        # Extract key fields from signal_dict
                        sig_str = signal_dict.get("signal", "neutral") if isinstance(signal_dict, dict) else "neutral"
                        strength = float(signal_dict.get("strength", 0.0)) if isinstance(signal_dict, dict) else 0.0
                        confidence = float(signal_dict.get("confidence", 0.5)) if isinstance(signal_dict, dict) else 0.5
                        explanation = signal_dict.get("explanation", "") if isinstance(signal_dict, dict) else ""
                        regime_context = signal_dict.get("regime_context", "") if isinstance(signal_dict, dict) else ""
                        
                        # Serialize non-signal details as metadata
                        meta = {}
                        for k, v in module_result.items():
                            if k != "signal":
                                try:
                                    json.dumps(v)
                                    meta[k] = v
                                except (TypeError, ValueError):
                                    meta[k] = str(v)
                        
                        self.pipeline.storage.save_module_signal(
                            module_name=name,
                            signal=sig_str,
                            strength=strength,
                            confidence=confidence,
                            explanation=str(explanation)[:500],
                            regime_context=str(regime_context)[:200],
                            regime_id=regime_result.regime,
                            metadata=meta if meta else None,
                        )
                    except Exception as persist_err:
                        logger.debug(f"Signal persistence for {name} skipped: {persist_err}")
            except Exception as e:
                logger.warning(f"Module {name} analysis failed: {e}")
                module_signals[name] = {"error": str(e)}
        
        results["modules"] = module_signals
        
        # Layer 3: Meta-learning checks
        if self.meta_learner:
            recalibration_decision = self.meta_learner.check_recalibration_needed()
            uncertainty_signal = self.meta_learner.generate_uncertainty_signal()
            adaptive_weights = self.meta_learner.get_adaptive_weights()
            
            # Push adaptive weights to ensemble
            if adaptive_weights and self.ensemble:
                self.ensemble.update_weights(adaptive_weights)
            
            # Execute recalibration if needed (pass ensemble + data for retraining)
            if recalibration_decision.should_recalibrate:
                self.meta_learner.execute_recalibration(
                    decision=recalibration_decision,
                    ensemble=self.ensemble,
                    training_data=data,
                )
            
            # Validate shadow mode if active
            shadow_result = self.meta_learner.validate_shadow_mode()
            
            results["meta"] = {
                "needs_recalibration": recalibration_decision.should_recalibrate,
                "recalibration_urgency": recalibration_decision.urgency_level,
                "recalibration_severity": recalibration_decision.severity,
                "recalibration_reasons": [str(r) for r in recalibration_decision.reasons],
                "uncertainty_signal": uncertainty_signal,
                "adaptive_weights": adaptive_weights,
                "shadow_validation": shadow_result,
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
        
        # Phase 2: Transition probabilities
        if self.transition_model and self.transition_model.is_fitted:
            try:
                forecast = self.transition_model.predict(
                    current_regime=regime_result.regime,
                    market_data=data,
                    disagreement=regime_result.disagreement,
                )
                results["transition_forecast"] = forecast.to_dict()
            except Exception as e:
                logger.warning(f"Transition forecast failed: {e}")
                results["transition_forecast"] = {"error": str(e)}
        
        # Phase 2: Multi-timeframe
        if self.multi_timeframe and self.multi_timeframe.is_fitted:
            try:
                mtf_result = self.multi_timeframe.predict(data)
                results["multi_timeframe"] = mtf_result.to_dict()
            except Exception as e:
                logger.warning(f"Multi-timeframe prediction failed: {e}")
                results["multi_timeframe"] = {"error": str(e)}
        
        # Phase 2: Narrative
        if self.narrative_generator:
            try:
                briefing = self.narrative_generator.generate(results)
                results["narrative"] = briefing.to_dict()
            except Exception as e:
                logger.warning(f"Narrative generation failed: {e}")
                results["narrative"] = {"error": str(e)}
        
        # Phase 3: Prediction Engine
        # Return forecasts
        if self.return_forecaster and self.return_forecaster.is_fitted:
            try:
                forecasts = self.return_forecaster.predict_all(
                    current_regime=regime_result.regime
                )
                results["return_forecasts"] = {
                    asset: fc.to_dict() for asset, fc in forecasts.items()
                }
            except Exception as e:
                logger.warning(f"Return forecast failed: {e}")
                results["return_forecasts"] = {"error": str(e)}
        
        # Tail risk analysis
        if self.tail_risk and self.tail_risk.is_fitted:
            try:
                # Default portfolio weights
                default_weights = {"SPX": 0.60, "TLT": 0.25, "GLD": 0.15}
                # Get transition probs from transition model if available
                trans_probs = None
                if "transition_forecast" in results and isinstance(
                    results["transition_forecast"], dict
                ):
                    bp = results["transition_forecast"].get("blended_probs")
                    if bp:
                        trans_probs = {int(k): float(v) for k, v in bp.items()}
                
                tail = self.tail_risk.analyze(
                    portfolio_weights=default_weights,
                    current_regime=regime_result.regime,
                    transition_probs=trans_probs,
                )
                results["tail_risk"] = tail.to_dict()
            except Exception as e:
                logger.warning(f"Tail risk analysis failed: {e}")
                results["tail_risk"] = {"error": str(e)}
        
        # Portfolio optimisation
        if self.portfolio_optimizer and self.portfolio_optimizer.is_fitted:
            try:
                trans_probs = None
                if "transition_forecast" in results and isinstance(
                    results["transition_forecast"], dict
                ):
                    bp = results["transition_forecast"].get("blended_probs")
                    if bp:
                        trans_probs = {int(k): float(v) for k, v in bp.items()}
                
                opt = self.portfolio_optimizer.optimize(
                    current_regime=regime_result.regime,
                    transition_probs=trans_probs,
                )
                results["portfolio_optimization"] = opt.to_dict()
            except Exception as e:
                logger.warning(f"Portfolio optimization failed: {e}")
                results["portfolio_optimization"] = {"error": str(e)}
        
        # Alpha signals
        if self.alpha_signals and self.alpha_signals.is_fitted:
            try:
                # Gather anomalies from correlation module
                corr_details = results.get("modules", {}).get(
                    "correlations", {}
                ).get("details", {})
                anomalies = corr_details.get(
                    "anomalies", corr_details.get("z_scores", {})
                )
                active = {}
                if isinstance(anomalies, dict):
                    for pair, val in anomalies.items():
                        if isinstance(val, (int, float)) and abs(val) > 0.1:
                            active[pair] = float(val)
                
                alpha = self.alpha_signals.generate(
                    current_regime=regime_result.regime,
                    active_anomalies=active,
                )
                results["alpha_signals"] = alpha.to_dict()
            except Exception as e:
                logger.warning(f"Alpha signal generation failed: {e}")
                results["alpha_signals"] = {"error": str(e)}
        
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
        
        # Count bullish/bearish/cautious signals
        bullish = 0
        bearish = 0
        cautious = 0
        
        for name, module in results["modules"].items():
            signal = module.get("signal", {})
            if isinstance(signal, dict):
                sig_val = signal.get("signal", "neutral")
                if sig_val == "bullish":
                    bullish += 1
                elif sig_val == "bearish":
                    bearish += 1
                elif sig_val == "cautious":
                    cautious += 1
        
        if bullish > bearish:
            summary["overall_bias"] = "Bullish"
        elif bearish > bullish:
            summary["overall_bias"] = "Bearish"
        elif cautious > 0:
            summary["overall_bias"] = "Cautious"
        else:
            summary["overall_bias"] = "Neutral"
        
        summary["signal_counts"] = {"bullish": bullish, "bearish": bearish, "cautious": cautious}
        
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
        """Recalibrate the system through the meta-learning layer."""
        if not self._is_initialized:
            raise RuntimeError("System not initialized")
        
        data = new_data if new_data is not None else self.market_data
        
        logger.info("Recalibrating AMRCAIS...")
        
        if self.meta_learner:
            # Use meta-learner path for proper tracking
            decision = self.meta_learner.check_recalibration_needed()
            # Force recalibration regardless of decision
            from src.meta_learning.recalibration import RecalibrationDecision, RecalibrationReason
            forced = RecalibrationDecision(
                should_recalibrate=True,
                reasons=[RecalibrationReason.HIGH_ERROR_RATE],
                severity=decision.severity if decision.should_recalibrate else 0.5,
                recommendations=["Manual recalibration requested"],
            )
            self.meta_learner.execute_recalibration(
                decision=forced,
                ensemble=self.ensemble,
                training_data=data,
            )
        else:
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
