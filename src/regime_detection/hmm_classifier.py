"""
Hidden Markov Model (HMM) Regime Classifier for AMRCAIS.

This module implements a Gaussian HMM-based regime classifier that identifies
market regimes based on the statistical properties of asset returns.

The HMM assumes markets transition between discrete hidden states (regimes)
where each state has distinct return distributions (mean and covariance).

Classes:
    HMMRegimeClassifier: 4-state Gaussian HMM for regime detection

Example:
    >>> classifier = HMMRegimeClassifier(n_states=4)
    >>> classifier.fit(returns_matrix)
    >>> result = classifier.predict(recent_returns)
    >>> print(f"Regime: {result.regime_name}, Confidence: {result.confidence:.2f}")
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import logging
import warnings

import pandas as pd
import numpy as np

from src.regime_detection.base import BaseClassifier, RegimeResult

logger = logging.getLogger(__name__)


class HMMRegimeClassifier(BaseClassifier):
    """Hidden Markov Model classifier for market regime detection.
    
    Uses a Gaussian HMM with 4 hidden states corresponding to:
    - State 1: Risk-On Growth (low vol, positive returns)
    - State 2: Risk-Off Crisis (high vol, negative returns, correlated)
    - State 3: Stagflation (moderate vol, mixed signals)
    - State 4: Disinflationary Boom (low vol, positive returns, falling rates)
    
    The classifier learns regime transition probabilities and emission
    distributions from historical multi-asset return data.
    
    Attributes:
        n_states: Number of hidden states (regimes)
        model: Trained GaussianHMM model
        state_mapping: Mapping from HMM states to AMRCAIS regimes
        
    Example:
        >>> # Prepare returns matrix
        >>> returns = pipeline.calculate_returns(data)
        >>> 
        >>> # Train HMM
        >>> hmm = HMMRegimeClassifier(n_states=4)
        >>> hmm.fit(returns)
        >>> 
        >>> # Predict current regime
        >>> recent = returns.iloc[-20:]
        >>> result = hmm.predict(recent)
        >>> print(f"Regime: {result.regime}, Confidence: {result.confidence:.2f}")
    """
    
    def __init__(
        self,
        n_states: int = 4,
        covariance_type: str = "full",
        n_iter: int = 100,
        tol: float = 0.01,
        random_state: int = 42,
        config: Optional[Dict] = None,
    ):
        """Initialize the HMM regime classifier.
        
        Args:
            n_states: Number of hidden states (should be 4 for AMRCAIS)
            covariance_type: Type of covariance ("full", "diag", "spherical")
            n_iter: Maximum EM iterations for training
            tol: Convergence threshold
            random_state: Random seed for reproducibility
            config: Optional configuration dictionary
        """
        super().__init__(n_regimes=n_states, name="HMM Classifier", config=config)
        
        self.n_states = n_states
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        
        self.model = None
        self.state_mapping: Dict[int, int] = {}
        self._feature_names: List[str] = []
        self._state_characteristics: Dict[int, Dict] = {}
    
    def _init_model(self):
        """Initialize the HMM model."""
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError(
                "hmmlearn is required for HMM classifier. "
                "Install with: pip install hmmlearn"
            )
        
        self.model = GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
    
    def fit(
        self,
        data: Union[pd.DataFrame, np.ndarray],
        labels: Optional[np.ndarray] = None,
    ) -> "HMMRegimeClassifier":
        """Train the HMM on historical return data.
        
        The HMM is unsupervised, so labels are only used for validating
        the learned states against known regime periods (optional).
        
        Args:
            data: Returns matrix (N observations × M assets)
                  Expected columns: SPX, TLT, GLD, VIX (at minimum)
            labels: Optional ground truth regimes for validation
            
        Returns:
            self: The fitted classifier
            
        Raises:
            ValueError: If data has insufficient samples or features
        """
        logger.info(f"Training HMM on {len(data)} observations")
        
        # Store feature names if DataFrame
        if isinstance(data, pd.DataFrame):
            self._feature_names = list(data.columns)
        
        # Validate and convert data
        X = self._validate_data(data, min_samples=252)  # At least 1 year
        
        # Initialize and fit model
        self._init_model()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model.fit(X)
        
        # Analyze learned states to map to AMRCAIS regimes
        self._analyze_states(X)
        
        # Map HMM states to AMRCAIS regime IDs
        self._map_states_to_regimes(X)
        
        self.is_fitted = True
        self._fit_timestamp = datetime.now()
        self._fit_samples = len(X)
        
        logger.info(
            f"HMM fitted with log-likelihood: {self.model.score(X):.2f}"
        )
        
        # Validate against labels if provided
        if labels is not None:
            self._validate_against_labels(X, labels)
        
        return self
    
    def _analyze_states(self, X: np.ndarray) -> None:
        """Analyze learned HMM states to characterize each regime.
        
        Examines the mean and covariance of each state to understand
        its characteristics (returns, volatility, correlations).
        """
        # Get state sequence
        states = self.model.predict(X)
        
        for state in range(self.n_states):
            mask = states == state
            state_data = X[mask]
            
            if len(state_data) == 0:
                continue
            
            # Calculate state characteristics
            mean_returns = np.mean(state_data, axis=0)
            volatility = np.std(state_data, axis=0)
            
            # If we have VIX (typically last column), use it for vol regime
            avg_vol = np.mean(volatility)
            
            # Calculate correlation structure
            if state_data.shape[1] > 1:
                corr_matrix = np.corrcoef(state_data.T)
                avg_correlation = np.mean(
                    corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
                )
            else:
                avg_correlation = 0
            
            self._state_characteristics[state] = {
                "mean_returns": mean_returns,
                "volatility": volatility,
                "avg_volatility": avg_vol,
                "avg_correlation": avg_correlation,
                "observation_count": len(state_data),
                "frequency": len(state_data) / len(X),
            }
            
            logger.debug(
                f"State {state}: avg_vol={avg_vol:.4f}, "
                f"avg_corr={avg_correlation:.2f}, freq={len(state_data)/len(X):.2%}"
            )
    
    def _map_states_to_regimes(self, X: np.ndarray) -> None:
        """Map HMM states to AMRCAIS regime IDs based on characteristics.
        
        Uses heuristics to match learned states to:
        1. Risk-On Growth: Low vol, positive returns
        2. Risk-Off Crisis: High vol, high correlations
        3. Stagflation: Moderate vol, mixed returns
        4. Disinflationary Boom: Low vol, positive returns, different from Risk-On
        
        This mapping is crucial for interpretability.
        """
        # Sort states by average volatility
        state_vols = [
            (state, char["avg_volatility"])
            for state, char in self._state_characteristics.items()
        ]
        state_vols.sort(key=lambda x: x[1])
        
        # Sort by correlation (for crisis detection)
        state_corrs = [
            (state, char["avg_correlation"])
            for state, char in self._state_characteristics.items()
        ]
        state_corrs.sort(key=lambda x: x[1], reverse=True)
        
        # Initialize mapping
        assigned = set()
        self.state_mapping = {}
        
        # Regime 2 (Risk-Off Crisis): Highest correlation + high volatility
        for state, corr in state_corrs:
            if state not in assigned:
                char = self._state_characteristics[state]
                if char["avg_volatility"] > np.median([c["avg_volatility"] for c in self._state_characteristics.values()]):
                    self.state_mapping[state] = 2
                    assigned.add(state)
                    break
        
        # Regime 1 (Risk-On Growth): Low volatility, not assigned
        for state, vol in state_vols:
            if state not in assigned:
                char = self._state_characteristics[state]
                # Check if returns are generally positive (first feature usually SPX)
                if char["mean_returns"][0] > 0:
                    self.state_mapping[state] = 1
                    assigned.add(state)
                    break
        
        # Regime 4 (Disinflationary Boom): Low-moderate vol, not assigned
        for state, vol in state_vols:
            if state not in assigned:
                self.state_mapping[state] = 4
                assigned.add(state)
                break
        
        # Regime 3 (Stagflation): Whatever is left
        for state in range(self.n_states):
            if state not in assigned:
                self.state_mapping[state] = 3
                assigned.add(state)
        
        # Log the mapping
        for hmm_state, regime in self.state_mapping.items():
            char = self._state_characteristics.get(hmm_state, {})
            logger.info(
                f"HMM State {hmm_state} → Regime {regime} ({self.REGIME_NAMES[regime]}): "
                f"vol={char.get('avg_volatility', 0):.4f}, "
                f"corr={char.get('avg_correlation', 0):.2f}"
            )
    
    def predict(
        self,
        data: Union[pd.DataFrame, np.ndarray, pd.Series],
    ) -> RegimeResult:
        """Predict the current regime from recent data.
        
        Uses the Viterbi algorithm to find the most likely state,
        then maps to AMRCAIS regime ID.
        
        Args:
            data: Recent returns data (single row or window for context)
            
        Returns:
            RegimeResult with regime, confidence, and state probabilities
            
        Raises:
            ValueError: If classifier not fitted or data invalid
        """
        if not self.is_fitted:
            raise ValueError("HMM classifier must be fitted before prediction")
        
        # Validate data
        X = self._validate_data(data, min_samples=1)
        
        # Get state probabilities
        log_prob = self.model.score(X)
        state_probs = self.model.predict_proba(X)
        
        # Get most likely state for the last observation
        if len(state_probs) > 1:
            # If multiple observations, use the last one
            final_probs = state_probs[-1]
        else:
            final_probs = state_probs[0]
        
        # Get predicted HMM state
        hmm_state = np.argmax(final_probs)
        confidence = final_probs[hmm_state]
        
        # Map to AMRCAIS regime
        regime = self.state_mapping.get(hmm_state, 1)
        
        # Build probability distribution over regimes
        regime_probs = {}
        for state, prob in enumerate(final_probs):
            mapped_regime = self.state_mapping.get(state, 1)
            if mapped_regime in regime_probs:
                regime_probs[mapped_regime] += prob
            else:
                regime_probs[mapped_regime] = prob
        
        result = RegimeResult(
            regime=regime,
            confidence=float(confidence),
            probabilities=regime_probs,
            features_used=self._feature_names,
            metadata={
                "hmm_state": int(hmm_state),
                "log_likelihood": float(log_prob),
                "state_characteristics": self._state_characteristics.get(hmm_state, {}),
            },
        )
        
        self._log_classification(result)
        return result
    
    def predict_proba(
        self,
        data: Union[pd.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """Get probability distribution over regimes.
        
        Args:
            data: Data for prediction
            
        Returns:
            Array of shape (N, 4) with regime probabilities
        """
        if not self.is_fitted:
            raise ValueError("HMM classifier must be fitted before prediction")
        
        X = self._validate_data(data, min_samples=1)
        state_probs = self.model.predict_proba(X)
        
        # Map HMM state probs to regime probs
        regime_probs = np.zeros((len(state_probs), self.n_regimes))
        
        for hmm_state, regime in self.state_mapping.items():
            regime_idx = regime - 1  # Convert 1-indexed to 0-indexed
            regime_probs[:, regime_idx] += state_probs[:, hmm_state]
        
        return regime_probs
    
    def get_transition_matrix(self) -> pd.DataFrame:
        """Get the regime transition probability matrix.
        
        Returns:
            DataFrame with transition probabilities between regimes
        """
        if not self.is_fitted:
            raise ValueError("HMM must be fitted first")
        
        # Get HMM transition matrix
        hmm_trans = self.model.transmat_
        
        # Map to regime transition matrix
        regime_trans = np.zeros((self.n_regimes, self.n_regimes))
        
        for from_state, from_regime in self.state_mapping.items():
            for to_state, to_regime in self.state_mapping.items():
                regime_trans[from_regime-1, to_regime-1] += (
                    hmm_trans[from_state, to_state]
                )
        
        # Normalize rows
        row_sums = regime_trans.sum(axis=1, keepdims=True)
        regime_trans = np.divide(
            regime_trans, row_sums,
            where=row_sums != 0
        )
        
        df = pd.DataFrame(
            regime_trans,
            index=[self.REGIME_NAMES[i] for i in range(1, 5)],
            columns=[self.REGIME_NAMES[i] for i in range(1, 5)],
        )
        
        return df
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """HMM doesn't have traditional feature importance.
        
        Returns the contribution of each feature to state separation
        based on the difference in means across states.
        """
        if not self.is_fitted or not self._feature_names:
            return None
        
        # Calculate feature importance as mean difference across states
        means = self.model.means_  # Shape: (n_states, n_features)
        
        importance = {}
        for i, feature in enumerate(self._feature_names):
            # Measure as range of means across states
            mean_range = np.max(means[:, i]) - np.min(means[:, i])
            importance[feature] = float(mean_range)
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        return importance
    
    def _validate_against_labels(
        self,
        X: np.ndarray,
        labels: np.ndarray,
    ) -> float:
        """Validate HMM predictions against ground truth labels.
        
        Args:
            X: Feature matrix
            labels: Ground truth regime labels (1-4)
            
        Returns:
            Accuracy score
        """
        predictions = self.model.predict(X)
        
        # Map predictions to regimes
        regime_predictions = np.array([
            self.state_mapping.get(p, 1) for p in predictions
        ])
        
        accuracy = np.mean(regime_predictions == labels)
        
        logger.info(f"HMM validation accuracy: {accuracy:.2%}")
        
        return accuracy
    
    def get_state_statistics(self) -> pd.DataFrame:
        """Get statistics for each learned state.
        
        Returns:
            DataFrame with state characteristics
        """
        if not self._state_characteristics:
            raise ValueError("HMM must be fitted first")
        
        records = []
        for state, char in self._state_characteristics.items():
            regime = self.state_mapping.get(state, 0)
            records.append({
                "HMM_State": state,
                "Regime": regime,
                "Regime_Name": self.REGIME_NAMES.get(regime, "Unknown"),
                "Avg_Volatility": char["avg_volatility"],
                "Avg_Correlation": char["avg_correlation"],
                "Observation_Count": char["observation_count"],
                "Frequency": char["frequency"],
            })
        
        return pd.DataFrame(records)
