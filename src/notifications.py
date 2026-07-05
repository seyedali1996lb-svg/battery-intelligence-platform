"""
Shared webhook sender for all proactive push alerts (IEC 62619 anomalies,
fleet digest, trajectory match, passport gap). Extracted from the
HMAC-signing block that used to live inline in the Live Monitor page and
the Settings "Send test ping" button — same behavior, now callable from
anywhere without touching session_state directly.
"""

import datetime
import hashlib
import hmac
import json

import requests


def send_webhook(
    event_type: str,
    payload: dict,
    webhook_url: str,
    webhook_secret: str | None = None,
    timeout: float = 3.0,
) -> bool:
    """
    POST a JSON payload to webhook_url, HMAC-SHA256 signed if webhook_secret
    is set (X-Signature-256 header, matching the existing anomaly-webhook
    convention). Returns True on a 2xx response, False otherwise — never
    raises, so callers can fire-and-forget without try/except.
    """
    if not webhook_url:
        return False

    body_dict = dict(payload)
    body_dict["event"] = event_type
    body_dict.setdefault("timestamp", datetime.datetime.now().isoformat())
    body_dict["source"] = "battery-intelligence-platform"
    body = json.dumps(body_dict).encode()

    headers = {"Content-Type": "application/json"}
    if webhook_secret:
        sig = hmac.new(webhook_secret.encode(), body, hashlib.sha256).hexdigest()
        headers["X-Signature-256"] = f"sha256={sig}"

    try:
        resp = requests.post(webhook_url, data=body, headers=headers, timeout=timeout)
        return resp.status_code < 300
    except Exception:
        return False
