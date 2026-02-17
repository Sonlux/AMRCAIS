"""
Multi-User + Collaboration — Phase 5.2.

Role-based access control with multi-tenant architecture.

User Roles:
    Researcher   → Full access, can create custom regime definitions
    PM           → View-only + allocation recommendations
    Risk Manager → Alerts + VaR decomposition
    CIO          → Executive dashboard with regime narrative

Classes:
    UserRole: Enum of supported roles.
    User: User account with role and permissions.
    UserManager: CRUD operations, authentication, and authorisation.

Example:
    >>> mgr = UserManager()
    >>> user = mgr.create_user("Alice", "alice@fund.com", UserRole.RESEARCHER)
    >>> mgr.authorize(user.user_id, "write_regime_definition")
    True
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


# ─── Roles & Permissions ─────────────────────────────────────────


class UserRole(str, Enum):
    """Supported user roles (Phase 5.2)."""

    RESEARCHER = "researcher"
    PM = "pm"
    RISK_MANAGER = "risk_manager"
    CIO = "cio"


# Default permission sets per role
_ROLE_PERMISSIONS: Dict[UserRole, Set[str]] = {
    UserRole.RESEARCHER: {
        "read_regime",
        "write_regime_definition",
        "read_modules",
        "write_annotations",
        "read_predictions",
        "write_predictions",
        "read_backtest",
        "write_backtest",
        "read_knowledge",
        "write_knowledge",
        "read_research",
        "write_research",
        "read_alt_data",
        "write_alt_data",
        "read_users",
        "manage_own_profile",
    },
    UserRole.PM: {
        "read_regime",
        "read_modules",
        "read_predictions",
        "read_backtest",
        "read_knowledge",
        "read_research",
        "read_alt_data",
        "read_allocations",
        "write_annotations",
        "manage_own_profile",
    },
    UserRole.RISK_MANAGER: {
        "read_regime",
        "read_modules",
        "read_predictions",
        "read_backtest",
        "read_knowledge",
        "read_research",
        "read_alt_data",
        "read_risk",
        "write_alerts",
        "read_var",
        "write_annotations",
        "manage_own_profile",
    },
    UserRole.CIO: {
        "read_regime",
        "read_modules",
        "read_predictions",
        "read_backtest",
        "read_knowledge",
        "read_research",
        "read_alt_data",
        "read_allocations",
        "read_risk",
        "read_var",
        "read_narrative",
        "read_users",
        "write_annotations",
        "manage_own_profile",
        "admin_dashboard",
    },
}


# ─── User Data ───────────────────────────────────────────────────


@dataclass
class User:
    """User account.

    Attributes:
        user_id: Unique identifier.
        name: Display name.
        email: Email address (unique).
        role: UserRole enum value.
        permissions: Set of permission strings.
        api_key_hash: SHA-256 hash of the API key (never store plaintext).
        created_at: Account creation timestamp.
        last_login: Last login timestamp.
        is_active: Whether account is active.
        preferences: User-specific settings.
        annotations: User's shared annotations.
    """

    user_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    name: str = ""
    email: str = ""
    role: UserRole = UserRole.PM
    permissions: Set[str] = field(default_factory=set)
    api_key_hash: str = ""
    created_at: str = field(
        default_factory=lambda: datetime.now().isoformat()
    )
    last_login: Optional[str] = None
    is_active: bool = True
    preferences: Dict[str, Any] = field(default_factory=dict)
    annotations: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self, include_sensitive: bool = False) -> Dict[str, Any]:
        """Serialize for API / JSON.

        Args:
            include_sensitive: Include API key hash.

        Returns:
            User as dict.
        """
        d: Dict[str, Any] = {
            "user_id": self.user_id,
            "name": self.name,
            "email": self.email,
            "role": self.role.value,
            "permissions": sorted(self.permissions),
            "created_at": self.created_at,
            "last_login": self.last_login,
            "is_active": self.is_active,
            "preferences": self.preferences,
            "annotation_count": len(self.annotations),
            "annotations": self.annotations,
        }
        if include_sensitive:
            d["api_key_hash"] = self.api_key_hash
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "User":
        """Deserialize from dict."""
        return cls(
            user_id=data.get("user_id", uuid.uuid4().hex[:12]),
            name=data.get("name", ""),
            email=data.get("email", ""),
            role=UserRole(data.get("role", "pm")),
            permissions=set(data.get("permissions", [])),
            api_key_hash=data.get("api_key_hash", ""),
            created_at=data.get("created_at", datetime.now().isoformat()),
            last_login=data.get("last_login"),
            is_active=data.get("is_active", True),
            preferences=data.get("preferences", {}),
            annotations=data.get("annotations", []),
        )


# ─── User Manager ───────────────────────────────────────────────


class UserManager:
    """Multi-user management engine with role-based access control.

    Supports the four role types required by Phase 5.2:
    Researcher, PM, Risk Manager, CIO.

    Args:
        storage_path: Path to the JSON user store.

    Example:
        >>> mgr = UserManager()
        >>> user, api_key = mgr.create_user("Alice", "alice@fund.com",
        ...                                  UserRole.RESEARCHER)
        >>> mgr.authorize(user.user_id, "write_regime_definition")
        True
    """

    def __init__(self, storage_path: str = "data/users.json") -> None:
        self._storage_path = Path(storage_path)
        self._users: Dict[str, User] = {}
        self._load()

    # ── CRUD ──────────────────────────────────────────────────

    def create_user(
        self,
        name: str,
        email: str,
        role: UserRole = UserRole.PM,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> tuple:
        """Create a new user account.

        Args:
            name: Display name.
            email: Email address (must be unique).
            role: User role.
            preferences: Optional user settings.

        Returns:
            Tuple of (User, api_key_plaintext).

        Raises:
            ValueError: If email already registered.
        """
        # Check email uniqueness
        for u in self._users.values():
            if u.email == email:
                raise ValueError(f"Email already registered: {email}")

        # Generate API key
        api_key = secrets.token_urlsafe(32)
        api_key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        # Build permission set from role defaults
        permissions = set(_ROLE_PERMISSIONS.get(role, set()))

        user = User(
            name=name,
            email=email,
            role=role,
            permissions=permissions,
            api_key_hash=api_key_hash,
            preferences=preferences or {},
        )

        self._users[user.user_id] = user
        self._save()

        logger.info(f"User created: {name} ({email}) role={role.value}")
        return user, api_key

    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID.

        Args:
            user_id: User identifier.

        Returns:
            User if found, None otherwise.
        """
        return self._users.get(user_id)

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email address.

        Args:
            email: Email to look up.

        Returns:
            User if found, None otherwise.
        """
        for u in self._users.values():
            if u.email == email:
                return u
        return None

    def list_users(
        self,
        role: Optional[UserRole] = None,
        active_only: bool = True,
    ) -> List[User]:
        """List all users, optionally filtered.

        Args:
            role: Filter by role.
            active_only: Only return active accounts.

        Returns:
            List of Users.
        """
        result = list(self._users.values())
        if role is not None:
            result = [u for u in result if u.role == role]
        if active_only:
            result = [u for u in result if u.is_active]
        return result

    def update_user(
        self,
        user_id: str,
        name: Optional[str] = None,
        role: Optional[UserRole] = None,
        is_active: Optional[bool] = None,
        preferences: Optional[Dict[str, Any]] = None,
    ) -> Optional[User]:
        """Update user fields.

        Args:
            user_id: User identifier.
            name: New display name.
            role: New role (permissions re-derived).
            is_active: Activate/deactivate.
            preferences: New preferences (merged).

        Returns:
            Updated User, or None if not found.
        """
        user = self._users.get(user_id)
        if not user:
            return None

        if name is not None:
            user.name = name
        if role is not None:
            user.role = role
            user.permissions = set(_ROLE_PERMISSIONS.get(role, set()))
        if is_active is not None:
            user.is_active = is_active
        if preferences is not None:
            user.preferences.update(preferences)

        self._save()
        return user

    def delete_user(self, user_id: str) -> bool:
        """Delete a user account.

        Args:
            user_id: User identifier.

        Returns:
            True if deleted, False if not found.
        """
        if user_id in self._users:
            del self._users[user_id]
            self._save()
            return True
        return False

    # ── Authentication ────────────────────────────────────────

    def authenticate(self, api_key: str) -> Optional[User]:
        """Authenticate a user by API key.

        Args:
            api_key: Plaintext API key to verify.

        Returns:
            User if authenticated, None otherwise.
        """
        key_hash = hashlib.sha256(api_key.encode()).hexdigest()

        for user in self._users.values():
            if user.api_key_hash == key_hash and user.is_active:
                user.last_login = datetime.now().isoformat()
                self._save()
                return user
        return None

    # ── Authorisation ─────────────────────────────────────────

    def authorize(self, user_id: str, permission: str) -> bool:
        """Check if a user has a specific permission.

        Args:
            user_id: User identifier.
            permission: Permission string to check.

        Returns:
            True if authorised, False otherwise.
        """
        user = self._users.get(user_id)
        if not user or not user.is_active:
            return False
        return permission in user.permissions

    def get_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for a user.

        Args:
            user_id: User identifier.

        Returns:
            Set of permission strings.
        """
        user = self._users.get(user_id)
        if not user:
            return set()
        return set(user.permissions)

    def grant_permission(self, user_id: str, permission: str) -> bool:
        """Grant an additional permission to a user.

        Args:
            user_id: User identifier.
            permission: Permission to grant.

        Returns:
            True if granted, False if user not found.
        """
        user = self._users.get(user_id)
        if not user:
            return False
        user.permissions.add(permission)
        self._save()
        return True

    def revoke_permission(self, user_id: str, permission: str) -> bool:
        """Revoke a permission from a user.

        Args:
            user_id: User identifier.
            permission: Permission to revoke.

        Returns:
            True if revoked, False if user not found.
        """
        user = self._users.get(user_id)
        if not user:
            return False
        user.permissions.discard(permission)
        self._save()
        return True

    # ── Annotations ───────────────────────────────────────────

    def add_annotation(
        self,
        user_id: str,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Add a shared annotation from a user.

        Args:
            user_id: Author's user ID.
            content: Annotation text.
            context: Additional context (regime, date, etc.).

        Returns:
            The annotation dict, or None if user not found.
        """
        user = self._users.get(user_id)
        if not user:
            return None

        annotation = {
            "annotation_id": uuid.uuid4().hex[:12],
            "author": user.name,
            "author_id": user_id,
            "role": user.role.value,
            "content": content,
            "context": context or {},
            "timestamp": datetime.now().isoformat(),
        }

        user.annotations.append(annotation)
        self._save()
        return annotation

    def get_annotations(
        self,
        user_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """Get annotations, optionally filtered by user.

        Args:
            user_id: Filter by author.
            limit: Maximum annotations to return.

        Returns:
            List of annotation dicts, most recent first.
        """
        all_notes: List[Dict[str, Any]] = []

        if user_id:
            user = self._users.get(user_id)
            if user:
                all_notes = user.annotations[:]
        else:
            for user in self._users.values():
                all_notes.extend(user.annotations)

        # Sort by timestamp desc
        all_notes.sort(key=lambda a: a.get("timestamp", ""), reverse=True)
        return all_notes[:limit]

    # ── Summary ───────────────────────────────────────────────

    def get_summary(self) -> Dict[str, Any]:
        """Get user manager status summary.

        Returns:
            Dict with user counts, role distribution, etc.
        """
        by_role: Dict[str, int] = {}
        active = 0
        total_annotations = 0

        for user in self._users.values():
            by_role[user.role.value] = by_role.get(user.role.value, 0) + 1
            if user.is_active:
                active += 1
            total_annotations += len(user.annotations)

        return {
            "total_users": len(self._users),
            "active_users": active,
            "by_role": by_role,
            "total_annotations": total_annotations,
            "roles_available": [r.value for r in UserRole],
        }

    # ── Persistence ───────────────────────────────────────────

    def _save(self) -> None:
        """Persist user data to JSON."""
        try:
            self._storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                uid: u.to_dict(include_sensitive=True)
                for uid, u in self._users.items()
            }
            self._storage_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            logger.error(f"User store save failed: {exc}")

    def _load(self) -> None:
        """Load user data from JSON."""
        if not self._storage_path.exists():
            return

        try:
            raw = json.loads(
                self._storage_path.read_text(encoding="utf-8")
            )
            self._users = {
                uid: User.from_dict(udata) for uid, udata in raw.items()
            }
            logger.info(f"Loaded {len(self._users)} users")
        except Exception as exc:
            logger.error(f"User store load failed: {exc}")
