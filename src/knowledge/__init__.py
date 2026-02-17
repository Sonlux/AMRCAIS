"""
AMRCAIS Knowledge Base — Phase 5: Network Effects + Moat.

Provides institutional memory that compounds with every analysis run:
- Regime transition indexing and pattern matching
- Anomaly catalog with outcome tracking
- Alternative data signal integration
- Research publication pipeline
- Multi-user role-based access control

Every analysis run feeds data into the knowledge base, making
AMRCAIS smarter over time — the key advantage over stateless
terminals like Bloomberg.

Classes:
    KnowledgeBase: Core institutional memory engine (5.1)
    AltDataIntegrator: Alternative data signal integration (5.3)
    ResearchPublisher: Auto-generated research reports (5.4)
    UserManager: Multi-user RBAC system (5.2)
"""

from src.knowledge.knowledge_base import (
    KnowledgeBase,
    RegimeTransitionRecord,
    AnomalyRecord,
    PatternMatch,
)
from src.knowledge.alt_data import AltDataIntegrator, AltDataSignal
from src.knowledge.research_publisher import ResearchPublisher, ResearchReport
from src.knowledge.user_manager import UserManager, User, UserRole

__all__ = [
    "KnowledgeBase",
    "RegimeTransitionRecord",
    "AnomalyRecord",
    "PatternMatch",
    "AltDataIntegrator",
    "AltDataSignal",
    "ResearchPublisher",
    "ResearchReport",
    "UserManager",
    "User",
    "UserRole",
]
