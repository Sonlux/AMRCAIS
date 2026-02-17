"""
Multi-channel alert delivery for AMRCAIS (Phase 4.2 — Delivery).

Implements concrete alert delivery channels:
    - Email (SMTP / SendGrid)
    - Slack (Incoming Webhook)
    - Telegram (Bot API)

Each channel is independently configured via environment variables
and can be enabled/disabled at runtime.

Configuration:
    # Email
    ALERT_EMAIL_ENABLED   — "true" | "false" (default: false)
    SMTP_HOST             — SMTP server hostname
    SMTP_PORT             — SMTP server port (default: 587)
    SMTP_USER             — SMTP username
    SMTP_PASSWORD         — SMTP password
    ALERT_EMAIL_FROM      — Sender address
    ALERT_EMAIL_TO        — Comma-separated recipient addresses

    # Slack
    ALERT_SLACK_ENABLED   — "true" | "false" (default: false)
    SLACK_WEBHOOK_URL     — Slack incoming webhook URL

    # Telegram
    ALERT_TELEGRAM_ENABLED — "true" | "false" (default: false)
    TELEGRAM_BOT_TOKEN     — Telegram bot token
    TELEGRAM_CHAT_ID       — Telegram chat/group ID

Classes:
    DeliveryChannel: Abstract base for delivery channels.
    EmailDelivery: SMTP email delivery.
    SlackDelivery: Slack webhook delivery.
    TelegramDelivery: Telegram bot delivery.
    AlertDeliveryManager: Orchestrates multi-channel delivery.

Example:
    >>> manager = AlertDeliveryManager.from_env()
    >>> manager.deliver(alert)
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
from abc import ABC, abstractmethod
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# Severity → display emoji
_SEVERITY_EMOJI = {
    "critical": "🔴",
    "high": "🟠",
    "medium": "🟡",
    "low": "🔵",
    "info": "⚪",
}


# ─── Abstract Channel ────────────────────────────────────────────


class DeliveryChannel(ABC):
    """Abstract base for alert delivery channels.

    Subclasses implement the `send()` method for their specific
    transport (email, webhook, bot API, etc.).
    """

    def __init__(self, name: str, enabled: bool = True) -> None:
        self.name = name
        self.enabled = enabled

    @abstractmethod
    def send(self, alert_dict: Dict[str, Any]) -> bool:
        """Deliver an alert through this channel.

        Args:
            alert_dict: Serialized alert (from Alert.to_dict()).

        Returns:
            True if delivery succeeded, False otherwise.
        """
        ...

    def _format_text(self, alert: Dict[str, Any]) -> str:
        """Format alert as plain text."""
        severity = alert.get("severity", "info")
        emoji = _SEVERITY_EMOJI.get(severity, "⚪")
        return (
            f"{emoji} [{severity.upper()}] {alert.get('title', 'Alert')}\n"
            f"{alert.get('message', '')}\n"
            f"Type: {alert.get('alert_type', 'unknown')}\n"
            f"ID: {alert.get('alert_id', 'N/A')}"
        )


# ─── Email Channel ───────────────────────────────────────────────


class EmailDelivery(DeliveryChannel):
    """SMTP email delivery channel.

    Args:
        smtp_host: SMTP server hostname.
        smtp_port: SMTP server port.
        smtp_user: SMTP authentication username.
        smtp_password: SMTP authentication password.
        from_addr: Sender email address.
        to_addrs: List of recipient email addresses.
        use_tls: Whether to use STARTTLS.

    Example:
        >>> email = EmailDelivery(
        ...     smtp_host="smtp.gmail.com",
        ...     from_addr="alerts@amrcais.io",
        ...     to_addrs=["trader@fund.com"],
        ... )
        >>> email.send(alert.to_dict())
    """

    def __init__(
        self,
        smtp_host: str = "localhost",
        smtp_port: int = 587,
        smtp_user: str = "",
        smtp_password: str = "",
        from_addr: str = "amrcais@localhost",
        to_addrs: Optional[List[str]] = None,
        use_tls: bool = True,
    ) -> None:
        super().__init__("email")
        self._host = smtp_host
        self._port = smtp_port
        self._user = smtp_user
        self._password = smtp_password
        self._from = from_addr
        self._to = to_addrs or []
        self._use_tls = use_tls

    def send(self, alert_dict: Dict[str, Any]) -> bool:
        """Send alert via SMTP email.

        Args:
            alert_dict: Serialized alert.

        Returns:
            True if sent successfully.
        """
        if not self._to:
            logger.warning("Email delivery: no recipients configured")
            return False

        try:
            severity = alert_dict.get("severity", "info")
            emoji = _SEVERITY_EMOJI.get(severity, "⚪")

            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"{emoji} AMRCAIS Alert: {alert_dict.get('title', 'Alert')}"
            msg["From"] = self._from
            msg["To"] = ", ".join(self._to)

            # Plain text
            text_body = self._format_text(alert_dict)
            msg.attach(MIMEText(text_body, "plain"))

            # HTML
            html_body = self._format_html(alert_dict)
            msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(self._host, self._port) as server:
                if self._use_tls:
                    server.starttls()
                if self._user and self._password:
                    server.login(self._user, self._password)
                server.sendmail(self._from, self._to, msg.as_string())

            logger.info(f"Email alert sent to {len(self._to)} recipients")
            return True

        except Exception as exc:
            logger.error(f"Email delivery failed: {exc}")
            return False

    def _format_html(self, alert: Dict[str, Any]) -> str:
        """Format alert as styled HTML email."""
        severity = alert.get("severity", "info")
        colors = {
            "critical": "#dc2626",
            "high": "#ea580c",
            "medium": "#ca8a04",
            "low": "#2563eb",
            "info": "#6b7280",
        }
        color = colors.get(severity, "#6b7280")

        return f"""
        <div style="font-family: 'Inter', Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <div style="background: {color}; color: white; padding: 16px 24px; border-radius: 8px 8px 0 0;">
                <h2 style="margin: 0;">{_SEVERITY_EMOJI.get(severity, '⚪')} {alert.get('title', 'Alert')}</h2>
                <p style="margin: 4px 0 0; opacity: 0.9; font-size: 14px;">{severity.upper()} | {alert.get('alert_type', 'unknown')}</p>
            </div>
            <div style="background: #1a1a2e; color: #e0e0e0; padding: 24px; border-radius: 0 0 8px 8px;">
                <p style="font-size: 16px; line-height: 1.6;">{alert.get('message', '')}</p>
                <hr style="border: 1px solid #333; margin: 16px 0;">
                <p style="font-size: 12px; color: #888;">
                    Alert ID: {alert.get('alert_id', 'N/A')} |
                    Generated by AMRCAIS
                </p>
            </div>
        </div>
        """


# ─── Slack Channel ───────────────────────────────────────────────


class SlackDelivery(DeliveryChannel):
    """Slack incoming webhook delivery channel.

    Args:
        webhook_url: Slack incoming webhook URL.

    Example:
        >>> slack = SlackDelivery("https://hooks.slack.com/services/T.../B.../...")
        >>> slack.send(alert.to_dict())
    """

    def __init__(self, webhook_url: str = "") -> None:
        super().__init__("slack")
        self._webhook_url = webhook_url

    def send(self, alert_dict: Dict[str, Any]) -> bool:
        """Send alert to Slack via incoming webhook.

        Args:
            alert_dict: Serialized alert.

        Returns:
            True if sent successfully.
        """
        if not self._webhook_url:
            logger.warning("Slack delivery: no webhook URL configured")
            return False

        try:
            import urllib.request

            severity = alert_dict.get("severity", "info")
            emoji = _SEVERITY_EMOJI.get(severity, "⚪")
            color = {
                "critical": "#dc2626",
                "high": "#ea580c",
                "medium": "#ca8a04",
                "low": "#2563eb",
                "info": "#6b7280",
            }.get(severity, "#6b7280")

            payload = {
                "attachments": [
                    {
                        "color": color,
                        "title": f"{emoji} {alert_dict.get('title', 'Alert')}",
                        "text": alert_dict.get("message", ""),
                        "fields": [
                            {"title": "Type", "value": alert_dict.get("alert_type", ""), "short": True},
                            {"title": "Severity", "value": severity.upper(), "short": True},
                        ],
                        "footer": f"AMRCAIS | Alert ID: {alert_dict.get('alert_id', 'N/A')}",
                    }
                ]
            }

            req = urllib.request.Request(
                self._webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)

            logger.info("Slack alert sent")
            return True

        except Exception as exc:
            logger.error(f"Slack delivery failed: {exc}")
            return False


# ─── Telegram Channel ────────────────────────────────────────────


class TelegramDelivery(DeliveryChannel):
    """Telegram Bot API delivery channel.

    Args:
        bot_token: Telegram bot token (from @BotFather).
        chat_id: Target chat/group/channel ID.

    Example:
        >>> tg = TelegramDelivery("123456:ABC-DEF...", "-100123456789")
        >>> tg.send(alert.to_dict())
    """

    def __init__(self, bot_token: str = "", chat_id: str = "") -> None:
        super().__init__("telegram")
        self._bot_token = bot_token
        self._chat_id = chat_id

    def send(self, alert_dict: Dict[str, Any]) -> bool:
        """Send alert via Telegram Bot API.

        Args:
            alert_dict: Serialized alert.

        Returns:
            True if sent successfully.
        """
        if not self._bot_token or not self._chat_id:
            logger.warning("Telegram delivery: bot token or chat ID not configured")
            return False

        try:
            import urllib.request

            text = self._format_text(alert_dict)
            url = (
                f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
            )
            payload = {
                "chat_id": self._chat_id,
                "text": text,
                "parse_mode": "HTML",
            }

            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)

            logger.info("Telegram alert sent")
            return True

        except Exception as exc:
            logger.error(f"Telegram delivery failed: {exc}")
            return False


# ─── Delivery Manager ────────────────────────────────────────────


class AlertDeliveryManager:
    """Orchestrates multi-channel alert delivery.

    Holds a list of configured DeliveryChannels and dispatches
    alerts to all enabled channels.  Failed deliveries are logged
    but do not block other channels.

    Args:
        channels: List of DeliveryChannel instances.

    Example:
        >>> manager = AlertDeliveryManager.from_env()
        >>> manager.deliver(alert.to_dict())
    """

    def __init__(self, channels: Optional[List[DeliveryChannel]] = None) -> None:
        self._channels: List[DeliveryChannel] = channels or []
        self._delivery_count = 0
        self._failure_count = 0

    def add_channel(self, channel: DeliveryChannel) -> None:
        """Add a delivery channel.

        Args:
            channel: DeliveryChannel instance to add.
        """
        self._channels.append(channel)
        logger.info(f"Added alert delivery channel: {channel.name}")

    def deliver(self, alert_dict: Dict[str, Any]) -> Dict[str, bool]:
        """Deliver an alert to all enabled channels.

        Args:
            alert_dict: Serialized alert (from Alert.to_dict()).

        Returns:
            Dict of channel_name → success boolean.
        """
        results: Dict[str, bool] = {}

        for channel in self._channels:
            if not channel.enabled:
                results[channel.name] = False
                continue

            success = channel.send(alert_dict)
            results[channel.name] = success
            self._delivery_count += 1
            if not success:
                self._failure_count += 1

        return results

    def get_status(self) -> Dict[str, Any]:
        """Get delivery manager status.

        Returns:
            Dict with channel states and counters.
        """
        return {
            "channels": [
                {"name": c.name, "enabled": c.enabled} for c in self._channels
            ],
            "total_deliveries": self._delivery_count,
            "total_failures": self._failure_count,
        }

    @staticmethod
    def from_env() -> "AlertDeliveryManager":
        """Create delivery manager from environment variables.

        Reads ALERT_*_ENABLED variables and configures channels
        accordingly.

        Returns:
            Configured AlertDeliveryManager.
        """
        manager = AlertDeliveryManager()

        # Email
        if os.getenv("ALERT_EMAIL_ENABLED", "false").lower() == "true":
            to_addrs = os.getenv("ALERT_EMAIL_TO", "").split(",")
            to_addrs = [a.strip() for a in to_addrs if a.strip()]
            channel = EmailDelivery(
                smtp_host=os.getenv("SMTP_HOST", "localhost"),
                smtp_port=int(os.getenv("SMTP_PORT", "587")),
                smtp_user=os.getenv("SMTP_USER", ""),
                smtp_password=os.getenv("SMTP_PASSWORD", ""),
                from_addr=os.getenv("ALERT_EMAIL_FROM", "amrcais@localhost"),
                to_addrs=to_addrs,
            )
            manager.add_channel(channel)

        # Slack
        if os.getenv("ALERT_SLACK_ENABLED", "false").lower() == "true":
            channel = SlackDelivery(
                webhook_url=os.getenv("SLACK_WEBHOOK_URL", ""),
            )
            manager.add_channel(channel)

        # Telegram
        if os.getenv("ALERT_TELEGRAM_ENABLED", "false").lower() == "true":
            channel = TelegramDelivery(
                bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
                chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
            )
            manager.add_channel(channel)

        return manager
