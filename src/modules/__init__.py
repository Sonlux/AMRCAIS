"""Analytical modules for AMRCAIS.

Contains regime-adaptive analytical modules:
- MacroEventTracker: Macro event impact analysis
- YieldCurveAnalyzer: Yield curve deformation analysis
- OptionsSurfaceMonitor: Options volatility surface analysis
- FactorExposureAnalyzer: Factor exposure and rotation
- CorrelationAnomalyDetector: Cross-asset correlation monitoring
"""

from src.modules.base import AnalyticalModule, ModuleSignal
from src.modules.macro_event_tracker import MacroEventTracker
from src.modules.yield_curve_analyzer import YieldCurveAnalyzer
from src.modules.options_surface_monitor import OptionsSurfaceMonitor
from src.modules.factor_exposure_analyzer import FactorExposureAnalyzer
from src.modules.correlation_anomaly_detector import CorrelationAnomalyDetector

__all__ = [
    "AnalyticalModule",
    "ModuleSignal",
    "MacroEventTracker",
    "YieldCurveAnalyzer",
    "OptionsSurfaceMonitor",
    "FactorExposureAnalyzer",
    "CorrelationAnomalyDetector",
]
