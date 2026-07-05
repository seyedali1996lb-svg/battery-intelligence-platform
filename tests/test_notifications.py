"""Unit tests for src/notifications.py — send_webhook()."""

import hashlib
import hmac
import json
import threading
import http.server

from notifications import send_webhook


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
