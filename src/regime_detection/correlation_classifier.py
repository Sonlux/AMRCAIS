"""
Correlation-Based Regime Classifier for AMRCAIS.

This module implements a clustering-based classifier that identifies regimes
based on the correlation structure of cross-asset returns. Different market
regimes exhibit characteristic correlation patterns:

- Risk-On: Negative equity-bond correlation, low cross-asset correlation
- Risk-Off: High positive correlations across all assets (everything sells)
- Stagflation: Positive equity-bond correlation, commodities decouple
- Disinflation: Mixed correlations, bonds/equities both positive

Classes:
    CorrelationRegimeClassifier: Clustering-based correlation regime detector

Example:
    >>> classifier = CorrelationRegimeClassifier()
    >>> classifier.fit(returns)
    >>> result = classifier.predict(recent_returns)
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import logging

import pandas as pd
import numpy as np
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.preprocessing import StandardScaler

from src.regime_detection.base import BaseClassifier, RegimeResult

logger = logging.getLogger(__name__)


class CorrelationRegimeClassifier(BaseClassifier):
    """Classifier based on cross-asset correlation structure.
    
    Uses clustering on rolling correlation matrices to identify
    distinct correlation regimes. The key insight is that different
    market environments produce characteristic correlation patterns.
    
    Key correlation pairs monitored:
    - SPX-TLT: Equity-bond correlation (negative in risk-on, positive in crisis)
    - SPX-GLD: Equity-gold (indicates safe-haven flows)
    - DXY-WTI: Dollar-oil (commodity pricing)
    - SPX-VIX: Equity-volatility (fear gauge)
    
    Attributes:
        correlation_window: Rolling window for correlation calculation
        cluster_model: Clustering model (Spectral or KMeans)
        regime_mapping: Mapping from cluster ID to AMRCAIS regime
        
    Example:
        >>> corr_classifier = CorrelationRegimeClassifier(window=60)
        >>> corr_classifier.fit(returns)
        >>> result = corr_classifier.predict(recent_returns)
    """
    
    # Key correlation pairs for regime detection
    CORRELATION_PAIRS = [
        ("SPX", "TLT"),
        ("SPX", "GLD"),
        ("SPX", "VIX"),
        ("DXY", "WTI"),
        ("TLT", "GLD"),
    ]
    
    # Expected correlation baselines by regime
    REGIME_BASELINES = {
        1: {  # Risk-On Growth
            ("SPX", "TLT"): -0.30,
            ("SPX", "GLD"): -0.10,
            ("SPX", "VIX"): -0.80,
            ("DXY", "WTI"): -0.40,
        },
        2: {  # Risk-Off Crisis
            ("SPX", "TLT"): 0.50,
            ("SPX", "GLD"): 0.60,
            ("SPX", "VIX"): -0.95,
            ("DXY", "WTI"): 0.20,
        },
        3: {  # Stagflation
            ("SPX", "TLT"): 0.40,
            ("SPX", "GLD"): 0.30,
            ("SPX", "VIX"): -0.70,
            ("DXY", "WTI"): -0.60,
        },
        4: {  # Disinflationary Boom
            ("SPX", "TLT"): 0.20,
            ("SPX", "GLD"): -0.20,
            ("SPX", "VIX"): -0.85,
            ("DXY", "WTI"): -0.30,
        },
    }
    
    def __init__(
        self,
        correlation_window: int = 60,
        n_clusters: int = 4,
        method: str = "kmeans",
        random_state: int = 42,
        config: Optional[Dict] = None,
    ):
        """Initialize the correlation regime classifier.
        
        Args:
            correlation_window: Rolling window for correlation calculation
            n_clusters: Number of clusters (should be 4)
            method: Clustering method ("kmeans" or "spectral")
            random_state: Random seed for reproducibility
            config: Optional configuration dictionary
        """
        super().__init__(n_regimes=n_clusters, name="Correlation Classifier", config=config)
        
        self.correlation_window = correlation_window
        self.n_clusters = n_clusters
        self.method = method
        self.random_state = random_state
        
        self.cluster_model = None
        self.scaler: Optional[StandardScaler] = None
        self.regime_mapping: Dict[int, int] = {}
        self._cluster_centers: Optional[np.ndarray] = None
        self._feature_names: List[str] = []
        self._assets: List[str] = []
    
    def _init_model(self) -> None:
        """Initialize the clustering model."""
        if self.method == "spectral":
            self.cluster_model = SpectralClustering(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                affinity="nearest_neighbors",
            )
        else:
            self.cluster_model = KMeans(
                n_clusters=self.n_clusters,
                random_state=self.random_state,
                n_init=10,
            )
        
        self.scaler = StandardScaler()
    
    def _calculate_correlation_features(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate rolling correlation features from returns.
        
        Args:
            data: Returns DataFrame with asset columns
            
        Returns:
            DataFrame with correlation features for each date
        """
        # Identify available assets
        available_assets = set(data.columns)
        self._assets = list(available_assets)
        
        # Calculate rolling correlations for available pairs
        features = {}
        
        for asset1, asset2 in self.CORRELATION_PAIRS:
            if asset1 in available_assets and asset2 in available_assets:
                corr = data[asset1].rolling(
                    self.correlation_window
                ).corr(data[asset2])
                
                feature_name = f"{asset1}_{asset2}_corr"
                features[feature_name] = corr
        
        # Also add average correlation (market regime indicator)
        if len(self._assets) > 1:
            rolling_corr_matrix = data.rolling(self.correlation_window).corr()
            
            # Calculate average pairwise correlation
            avg_corr = []
            for date in data.index:
                try:
                    # Get correlation matrix for this date
                    corr_slice = rolling_corr_matrix.loc[date]
                    if isinstance(corr_slice, pd.DataFrame):
                        # Get upper triangle (excluding diagonal)
                        n = len(self._assets)
                        upper_tri = []
                        for i in range(n):
                            for j in range(i+1, n):
                                val = corr_slice.iloc[i, j]
                                if not np.isnan(val):
                                    upper_tri.append(val)
                        avg_corr.append(np.mean(upper_tri) if upper_tri else np.nan)
                    else:
                        avg_corr.append(np.nan)
                except (KeyError, IndexError):
                    avg_corr.append(np.nan)
            
            features["avg_correlation"] = avg_corr
        
        feature_df = pd.DataFrame(features, index=data.index)
        self._feature_names = list(feature_df.columns)
        
        return feature_df
    
    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        labels: Optional[np.ndarray] = None,
    ) -> "CorrelationRegimeClassifier":
        """Train the correlation-based classifier.
        
        Args:
            data: Returns DataFrame with asset columns
            labels: Optional labels for validation
            
        Returns:
            self: The fitted classifier
        """
        logger.info(f"Training correlation classifier on {len(data)} observations")
        
        if isinstance(data, np.ndarray):
            # Assume standard asset order
            columns = ["SPX", "TLT", "GLD", "DXY", "WTI", "VIX"][:data.shape[1]]
            data = pd.DataFrame(data, columns=columns)
        
        # Calculate correlation features
        corr_features = self._calculate_correlation_features(data)
        corr_features = corr_features.dropna()
        
        if len(corr_features) < 100:
            raise ValueError(
                f"Insufficient data after correlation calculation: "
                f"{len(corr_features)} samples"
            )
        
        # Initialize model
        self._init_model()
        
        # Scale features
        X = corr_features.values
        # Handle NaN/inf values before scaling
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit clustering
        cluster_labels = self.cluster_model.fit_predict(X_scaled)
        
        # Store cluster centers (for KMeans)
        if hasattr(self.cluster_model, "cluster_centers_"):
            self._cluster_centers = self.scaler.inverse_transform(
                self.cluster_model.cluster_centers_
            )
        
        # Map clusters to regimes based on correlation characteristics
        self._map_clusters_to_regimes(corr_features, cluster_labels)
        
        self.is_fitted = True
        self._fit_timestamp = datetime.now()
        self._fit_samples = len(corr_features)
        
        logger.info(
            f"Correlation classifier fitted with {self.n_clusters} clusters"
        )
        
        return self
    
    def _map_clusters_to_regimes(
        self,
        features: pd.DataFrame,
        cluster_labels: np.ndarray,
    ) -> None:
        """Map cluster IDs to AMRCAIS regime IDs based on correlation patterns."""
        # Calculate average correlations for each cluster
        cluster_stats = {}
        
        for cluster in range(self.n_clusters):
            mask = cluster_labels == cluster
            cluster_data = features[mask]
            
            if len(cluster_data) == 0:
                continue
            
            cluster_stats[cluster] = {
                col: cluster_data[col].mean()
                for col in features.columns
            }
        
        # Match clusters to regimes based on similarity to baselines
        assigned_regimes = set()
        self.regime_mapping = {}
        
        for cluster, stats in cluster_stats.items():
            best_regime = None
            best_score = float("inf")
            
            for regime in range(1, 5):
                if regime in assigned_regimes:
                    continue
                
                # Calculate distance to regime baseline
                score = 0
                count = 0
                
                for (a1, a2), baseline in self.REGIME_BASELINES.get(regime, {}).items():
                    col = f"{a1}_{a2}_corr"
                    if col in stats:
                        score += (stats[col] - baseline) ** 2
                        count += 1
                
                if count > 0:
                    score = score / count
                    if score < best_score:
                        best_score = score
                        best_regime = regime
            
            if best_regime is not None:
                self.regime_mapping[cluster] = best_regime
                assigned_regimes.add(best_regime)
                logger.debug(
                    f"Cluster {cluster} → Regime {best_regime} "
                    f"({self.REGIME_NAMES[best_regime]})"
                )
        
        # Assign remaining regimes to unmapped clusters
        for cluster in range(self.n_clusters):
            if cluster not in self.regime_mapping:
                for regime in range(1, 5):
                    if regime not in assigned_regimes:
                        self.regime_mapping[cluster] = regime
                        assigned_regimes.add(regime)
                        break
                else:
                    # Fallback
                    self.regime_mapping[cluster] = 1
    
    def predict(
        self,
        data: Union[pd.DataFrame, np.ndarray, pd.Series],
    ) -> RegimeResult:
        """Predict regime from recent returns data.
        
        Args:
            data: Recent returns (should include correlation_window rows)
            
        Returns:
            RegimeResult with regime and confidence
        """
        if not self.is_fitted:
            raise ValueError("Classifier must be fitted before prediction")
        
        # Convert to DataFrame if needed
        if isinstance(data, np.ndarray):
            columns = ["SPX", "TLT", "GLD", "DXY", "WTI", "VIX"][:data.shape[1]]
            data = pd.DataFrame(data, columns=columns)
        elif isinstance(data, pd.Series):
            data = data.to_frame().T
        
        # Need enough data for correlation calculation
        if len(data) < self.correlation_window:
            logger.warning(
                f"Insufficient data ({len(data)} rows) for correlation window "
                f"({self.correlation_window}). Using available data."
            )
        
        # Calculate correlation features
        corr_features = self._calculate_correlation_features(data)
        corr_features = corr_features.dropna()
        
        if len(corr_features) == 0:
            # Fall back to direct correlation calculation
            corr_features = self._calculate_direct_correlations(data)
        
        # Use last row
        X = corr_features.iloc[[-1]].values
        # Handle NaN/inf values before scaling
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        X_scaled = self.scaler.transform(X)
        
        # Predict cluster
        if self.method == "kmeans":
            cluster = self.cluster_model.predict(X_scaled)[0]
            
            # Calculate distance-based confidence
            distances = np.linalg.norm(
                self.cluster_model.cluster_centers_ - X_scaled,
                axis=1
            )
            confidence = 1.0 / (1.0 + distances[cluster])
        else:
            # For spectral clustering, find nearest cluster center
            if self._cluster_centers is not None:
                distances = np.linalg.norm(
                    self._cluster_centers - X,
                    axis=1
                )
                cluster = np.argmin(distances)
                confidence = 1.0 / (1.0 + distances[cluster])
            else:
                cluster = 0
                confidence = 0.5
        
        # Map to regime
        regime = self.regime_mapping.get(cluster, 1)
        
        # Calculate regime probabilities based on distances
        probabilities = self._calculate_regime_probabilities(X_scaled)
        
        result = RegimeResult(
            regime=regime,
            confidence=float(confidence),
            probabilities=probabilities,
            features_used=self._feature_names,
            metadata={
                "cluster": int(cluster),
                "correlation_features": {
                    name: float(corr_features.iloc[-1][name])
                    for name in self._feature_names
                    if name in corr_features.columns
                },
            },
        )
        
        self._log_classification(result)
        return result
    
    def _calculate_direct_correlations(
        self,
        data: pd.DataFrame,
    ) -> pd.DataFrame:
        """Calculate correlations directly when insufficient rolling window."""
        features = {}
        
        for asset1, asset2 in self.CORRELATION_PAIRS:
            if asset1 in data.columns and asset2 in data.columns:
                corr = data[asset1].corr(data[asset2])
                feature_name = f"{asset1}_{asset2}_corr"
                features[feature_name] = [corr]
        
        # Average correlation
        if len(data.columns) > 1:
            corr_matrix = data.corr()
            upper_tri = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
            features["avg_correlation"] = [np.mean(upper_tri)]
        
        return pd.DataFrame(features)
    
    def _calculate_regime_probabilities(
        self,
        X_scaled: np.ndarray,
    ) -> Dict[int, float]:
        """Calculate probability distribution over regimes."""
        probabilities = {r: 0.0 for r in range(1, 5)}
        
        if self.method == "kmeans" and hasattr(self.cluster_model, "cluster_centers_"):
            # Use softmax of negative distances
            distances = np.linalg.norm(
                self.cluster_model.cluster_centers_ - X_scaled,
                axis=1
            )
            
            # Convert to probabilities via softmax
            exp_neg_dist = np.exp(-distances)
            probs = exp_neg_dist / exp_neg_dist.sum()
            
            for cluster, prob in enumerate(probs):
                regime = self.regime_mapping.get(cluster, 1)
                probabilities[regime] += prob
        else:
            # Uniform fallback
            for r in range(1, 5):
                probabilities[r] = 0.25
        
        return probabilities
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance based on cluster separation.
        
        Returns variance of each feature across cluster centers.
        """
        if not self.is_fitted or self._cluster_centers is None:
            return None
        
        # Variance across cluster centers
        variances = np.var(self._cluster_centers, axis=0)
        
        importance = dict(zip(self._feature_names, variances))
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        return importance
    
    def get_correlation_baselines(self) -> pd.DataFrame:
        """Get learned correlation baselines for each regime.
        
        Returns:
            DataFrame with correlation features for each regime
        """
        if self._cluster_centers is None:
            raise ValueError("Classifier must be fitted with KMeans")
        
        records = []
        for cluster, center in enumerate(self._cluster_centers):
            regime = self.regime_mapping.get(cluster, cluster + 1)
            record = {
                "Regime": regime,
                "Regime_Name": self.REGIME_NAMES.get(regime, f"Regime {regime}"),
            }
            for i, feature in enumerate(self._feature_names):
                record[feature] = center[i]
            records.append(record)
        
        df = pd.DataFrame(records)
        df = df.sort_values("Regime").reset_index(drop=True)
        
        return df
