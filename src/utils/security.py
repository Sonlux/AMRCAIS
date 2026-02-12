"""
Security utilities for AMRCAIS.

This module provides security features including:
- API key validation and secure loading
- Rate limiting
- Input sanitization
- Secure configuration management

Classes:
    APIKeyManager: Secure API key management
    RateLimiter: Rate limiting for external API calls
    SecurityValidator: Input validation and sanitization
"""

import os
import re
import time
from typing import Optional, Dict, Any
from pathlib import Path
import logging
from datetime import datetime, timedelta
from collections import defaultdict

logger = logging.getLogger(__name__)


class APIKeyManager:
    """Secure API key management with validation.
    
    Loads API keys from environment variables and validates their format.
    Never exposes keys in logs or error messages.
    
    Example:
        >>> manager = APIKeyManager()
        >>> fred_key = manager.get_api_key("FRED_API_KEY")
        >>> if fred_key:
        ...     # Use key securely
        ...     pass
    """
    
    # Expected key patterns for validation
    KEY_PATTERNS = {
        "FRED_API_KEY": r'^[a-f0-9]{32}$',  # 32 hex chars
        "ALPHAVANTAGE_API_KEY": r'^[A-Z0-9]{16}$',  # 16 uppercase alphanumeric
    }
    
    def __init__(self):
        """Initialize API key manager."""
        self._keys: Dict[str, Optional[str]] = {}
        self._load_keys()
    
    def _load_keys(self) -> None:
        """Load API keys from environment variables."""
        for key_name in self.KEY_PATTERNS.keys():
            value = os.getenv(key_name)
            if value and value != f"your_{key_name.lower()}_here":
                self._keys[key_name] = value
                logger.info(f"Loaded API key: {key_name} (validated: {self._validate_key(key_name, value)})")
            else:
                self._keys[key_name] = None
                logger.warning(f"API key not found: {key_name}")
    
    def _validate_key(self, key_name: str, key_value: str) -> bool:
        """Validate API key format.
        
        Args:
            key_name: Name of the key
            key_value: Key value to validate
            
        Returns:
            True if key format is valid
        """
        if key_name not in self.KEY_PATTERNS:
            return True  # Unknown key type, assume valid
        
        pattern = self.KEY_PATTERNS[key_name]
        return bool(re.match(pattern, key_value))
    
    def get_api_key(self, key_name: str) -> Optional[str]:
        """Get API key securely.
        
        Args:
            key_name: Name of the API key (e.g., "FRED_API_KEY")
            
        Returns:
            API key value or None if not found
        """
        key = self._keys.get(key_name)
        
        if key is None:
            logger.error(
                f"API key '{key_name}' not configured. "
                f"Set environment variable or add to .env file."
            )
        
        return key
    
    def has_key(self, key_name: str) -> bool:
        """Check if API key is configured.
        
        Args:
            key_name: Name of the API key
            
        Returns:
            True if key is configured and valid
        """
        return self._keys.get(key_name) is not None
    
    def mask_key(self, key_value: str) -> str:
        """Mask API key for safe logging.
        
        Args:
            key_value: API key to mask
            
        Returns:
            Masked key (e.g., "abcd****xyz")
        """
        if not key_value or len(key_value) < 8:
            return "****"
        
        return f"{key_value[:4]}****{key_value[-4:]}"


class RateLimiter:
    """Rate limiting for external API calls.
    
    Prevents exceeding API rate limits by tracking request counts
    and enforcing delays when necessary.
    
    Example:
        >>> limiter = RateLimiter(max_requests=60, window_seconds=60)
        >>> limiter.wait_if_needed("fred_api")
        >>> # Make API call
    """
    
    def __init__(
        self,
        max_requests: int = 60,
        window_seconds: int = 60,
    ):
        """Initialize rate limiter.
        
        Args:
            max_requests: Maximum requests per window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        
        # Track requests per endpoint
        self._request_times: Dict[str, list] = defaultdict(list)
        
        logger.info(
            f"RateLimiter initialized: {max_requests} requests per {window_seconds}s"
        )
    
    def wait_if_needed(self, endpoint: str) -> None:
        """Wait if rate limit would be exceeded.
        
        Args:
            endpoint: Endpoint identifier for tracking
        """
        now = time.time()
        
        # Remove old requests outside window
        cutoff = now - self.window_seconds
        self._request_times[endpoint] = [
            t for t in self._request_times[endpoint]
            if t > cutoff
        ]
        
        # Check if at limit
        if len(self._request_times[endpoint]) >= self.max_requests:
            # Calculate wait time
            oldest_request = min(self._request_times[endpoint])
            wait_time = self.window_seconds - (now - oldest_request)
            
            if wait_time > 0:
                logger.warning(
                    f"Rate limit reached for {endpoint}. "
                    f"Waiting {wait_time:.1f}s"
                )
                time.sleep(wait_time + 0.5)  # Add small buffer
        
        # Record this request
        self._request_times[endpoint].append(time.time())
    
    def get_remaining_requests(self, endpoint: str) -> int:
        """Get remaining requests in current window.
        
        Args:
            endpoint: Endpoint to check
            
        Returns:
            Number of remaining requests
        """
        now = time.time()
        cutoff = now - self.window_seconds
        
        recent_requests = [
            t for t in self._request_times[endpoint]
            if t > cutoff
        ]
        
        return max(0, self.max_requests - len(recent_requests))


class SecurityValidator:
    """Input validation and sanitization.
    
    Validates and sanitizes user inputs to prevent injection attacks
    and ensure data integrity.
    
    Example:
        >>> validator = SecurityValidator()
        >>> safe_symbol = validator.sanitize_symbol("SPX'; DROP TABLE--")
        >>> # safe_symbol = "SPX"
    """
    
    # Allowed characters for different input types
    SYMBOL_PATTERN = re.compile(r'^[A-Z0-9\.\-\_]{1,10}$')
    DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    
    @staticmethod
    def sanitize_symbol(symbol: str) -> str:
        """Sanitize ticker symbol input.
        
        Args:
            symbol: Raw symbol input
            
        Returns:
            Sanitized symbol (uppercase, alphanumeric + common chars)
        """
        # Convert to uppercase and remove whitespace
        symbol = symbol.strip().upper()
        
        # Keep only allowed characters
        sanitized = re.sub(r'[^A-Z0-9\.\-\_]', '', symbol)
        
        # Limit length
        sanitized = sanitized[:10]
        
        if sanitized != symbol:
            logger.warning(
                f"Symbol sanitized: '{symbol}' → '{sanitized}'"
            )
        
        return sanitized
    
    @staticmethod
    def validate_date(date_str: str) -> bool:
        """Validate date string format.
        
        Args:
            date_str: Date string to validate
            
        Returns:
            True if valid YYYY-MM-DD format
        """
        if not SecurityValidator.DATE_PATTERN.match(date_str):
            return False
        
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    
    @staticmethod
    def validate_regime(regime: int) -> bool:
        """Validate regime value.
        
        Args:
            regime: Regime identifier
            
        Returns:
            True if regime in valid range (1-4)
        """
        return isinstance(regime, int) and 1 <= regime <= 4
    
    @staticmethod
    def sanitize_path(path: str) -> Path:
        """Sanitize file path to prevent directory traversal.
        
        Args:
            path: File path to sanitize
            
        Returns:
            Sanitized Path object
        """
        # Convert to Path object
        path_obj = Path(path).resolve()
        
        # Ensure path is within project directory
        project_root = Path(__file__).parent.parent.parent.resolve()
        
        try:
            path_obj.relative_to(project_root)
            return path_obj
        except ValueError:
            logger.error(
                f"Path '{path}' outside project directory. "
                "Potential directory traversal attack."
            )
            raise ValueError(f"Invalid path: {path}")
    
    @staticmethod
    def validate_config_value(
        value: Any,
        expected_type: type,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
    ) -> bool:
        """Validate configuration value.
        
        Args:
            value: Value to validate
            expected_type: Expected Python type
            min_value: Minimum value (for numeric types)
            max_value: Maximum value (for numeric types)
            
        Returns:
            True if valid
        """
        # Type check
        if not isinstance(value, expected_type):
            logger.error(
                f"Invalid type: expected {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )
            return False
        
        # Range check for numeric types
        if isinstance(value, (int, float)):
            if min_value is not None and value < min_value:
                logger.error(f"Value {value} below minimum {min_value}")
                return False
            
            if max_value is not None and value > max_value:
                logger.error(f"Value {value} above maximum {max_value}")
                return False
        
        return True


class SecureConfigLoader:
    """Secure configuration loading with validation.
    
    Loads YAML configuration files with security validation to prevent
    malicious config injection.
    
    Example:
        >>> loader = SecureConfigLoader()
        >>> config = loader.load_config("config/regimes.yaml")
    """
    
    def __init__(self):
        """Initialize secure config loader."""
        self.validator = SecurityValidator()
    
    def load_config(self, config_path: str) -> Dict:
        """Load and validate configuration file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Validated configuration dictionary
        """
        import yaml
        
        # Sanitize path
        safe_path = self.validator.sanitize_path(config_path)
        
        if not safe_path.exists():
            raise FileNotFoundError(f"Config file not found: {safe_path}")
        
        # Load YAML safely
        try:
            with open(safe_path, 'r') as f:
                config = yaml.safe_load(f)
            
            logger.info(f"Loaded configuration from {safe_path}")
            return config
        
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in {safe_path}: {e}")
            raise ValueError(f"Config file corrupted: {e}")
        
        except Exception as e:
            logger.error(f"Failed to load config {safe_path}: {e}")
            raise


# Global instances for easy access
_api_key_manager = None
_rate_limiter = None


def get_api_key_manager() -> APIKeyManager:
    """Get global API key manager instance."""
    global _api_key_manager
    if _api_key_manager is None:
        _api_key_manager = APIKeyManager()
    return _api_key_manager


def get_rate_limiter() -> RateLimiter:
    """Get global rate limiter instance."""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter
