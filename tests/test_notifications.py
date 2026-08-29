"""Unit tests for src/notifications.py — send_webhook() and notify_subscribers()."""

import hashlib
import hmac
import json
import sys
import threading
import http.server
import pathlib

import sys as _sys
import os as _os
_root = _os.path.abspath(_os.path.join(_os.path.dirname(__file__), ".."))
if _root not in _sys.path:
    _sys.path.insert(0, _root)
import _paths  # noqa: F401
from notifications import send_webhook, notify_subscribers


def test_returns_false_with_no_url():
    assert send_webhook("TEST", {"a": 1}, "", None) is False


def test_returns_false_on_unreachable_host():
    assert send_webhook("TEST", {"a": 1}, "http://127.0.0.1:1/nowhere", None, timeout=1) is False


def test_delivers_correct_payload_and_signature():
    received = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            received.append({"sig": self.headers.get("X-Signature-256"), "body": json.loads(body)})
            self.send_response(200)
            self.end_headers()
        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        ok = send_webhook(
            "TRAJECTORY_MATCH", {"cell_id": "Cell1", "warning_level": "high"},
            f"http://127.0.0.1:{port}/hook", "test-secret",
        )
        assert ok is True
        assert len(received) == 1
        body = received[0]["body"]
        assert body["event"] == "TRAJECTORY_MATCH"
        assert body["cell_id"] == "Cell1"
        assert body["source"] == "battery-intelligence-platform"
        assert "timestamp" in body

        expected_sig = "sha256=" + hmac.new(
            b"test-secret", json.dumps(body).encode(), hashlib.sha256
        ).hexdigest()
        assert received[0]["sig"] == expected_sig
    finally:
        server.shutdown()


def test_no_signature_header_without_secret():
    received = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            self.rfile.read(length)
            received.append(self.headers.get("X-Signature-256"))
            self.send_response(200)
            self.end_headers()
        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        send_webhook("FLEET_DIGEST", {"n_cells": 5}, f"http://127.0.0.1:{port}/hook", None)
        assert received == [None]
    finally:
        server.shutdown()


# ---------------------------------------------------------------------------
# notify_subscribers() — additional webhook destinations, additive to
# send_webhook()/the single legacy webhook_url setting above.
# ---------------------------------------------------------------------------

def _fake_subscriptions(monkeypatch, subs: list):
    import db as db_module
    monkeypatch.setattr(db_module, "get_webhook_subscriptions", lambda org_id: subs)


def test_notify_subscribers_returns_empty_list_with_no_subscriptions(monkeypatch):
    _fake_subscriptions(monkeypatch, [])
    assert notify_subscribers(1, "FLEET_DIGEST", {"n_cells": 5}) == []


def test_notify_subscribers_filters_by_event_types():
    """A subscription scoped to specific event_types must not fire for an
    event outside that list."""
    received = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            received.append(body["event"])
            self.send_response(200)
            self.end_headers()
        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        import db as db_module
        subs = [{
            "id": "wh-1", "name": "Only Digests", "url": f"http://127.0.0.1:{port}/hook",
            "secret": None, "event_types": ["FLEET_DIGEST"],
        }]
        orig = db_module.get_webhook_subscriptions
        db_module.get_webhook_subscriptions = lambda org_id: subs
        try:
            notify_subscribers(1, "FLEET_DIGEST", {"n_cells": 5})
            notify_subscribers(1, "TRAJECTORY_MATCH", {"cell_id": "X"})
        finally:
            db_module.get_webhook_subscriptions = orig
        assert received == ["FLEET_DIGEST"]
    finally:
        server.shutdown()


def test_notify_subscribers_empty_event_types_means_all_events():
    received = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            received.append(body["event"])
            self.send_response(200)
            self.end_headers()
        def log_message(self, *a):
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        import db as db_module
        subs = [{
            "id": "wh-1", "name": "Everything", "url": f"http://127.0.0.1:{port}/hook",
            "secret": None, "event_types": [],
        }]
        orig = db_module.get_webhook_subscriptions
        db_module.get_webhook_subscriptions = lambda org_id: subs
        try:
            notify_subscribers(1, "SOME_NOVEL_EVENT_TYPE", {"x": 1})
        finally:
            db_module.get_webhook_subscriptions = orig
        assert received == ["SOME_NOVEL_EVENT_TYPE"]
    finally:
        server.shutdown()


def test_notify_subscribers_reports_per_subscription_result(monkeypatch):
    import db as db_module
    subs = [
        {"id": "wh-1", "name": "Bad Host", "url": "http://127.0.0.1:1/nowhere", "secret": None, "event_types": []},
    ]
    monkeypatch.setattr(db_module, "get_webhook_subscriptions", lambda org_id: subs)
    results = notify_subscribers(1, "TEST", {"a": 1}, timeout=1)
    assert results == [{"subscription_id": "wh-1", "name": "Bad Host", "ok": False}]


def test_notify_subscribers_never_raises_when_db_lookup_fails(monkeypatch):
    import db as db_module

    def _boom(org_id):
        raise RuntimeError("simulated DB failure")

    monkeypatch.setattr(db_module, "get_webhook_subscriptions", _boom)
    assert notify_subscribers(1, "TEST", {"a": 1}) == []
