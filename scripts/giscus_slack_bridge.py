#!/usr/bin/env python3
"""Forward Giscus comment webhooks to Slack."""
from __future__ import annotations
import os
import hmac
import hashlib
import logging
from flask import Flask, request, abort
import requests

SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")
GITHUB_SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET")

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)


def verify(sig: str, payload: bytes) -> bool:
    if not GITHUB_SECRET or not sig:
        return True
    digest = hmac.new(GITHUB_SECRET.encode(), payload, hashlib.sha256).hexdigest()
    return hmac.compare_digest(f"sha256={digest}", sig)


@app.post("/webhook")
def handle_webhook():
    signature = request.headers.get("X-Hub-Signature-256", "")
    if not verify(signature, request.data):
        abort(400)

    if request.headers.get("X-GitHub-Event") != "discussion_comment":
        return "", 204

    payload = request.get_json(force=True)
    comment = payload.get("comment", {})
    user = comment.get("user", {}).get("login", "unknown")
    url = comment.get("html_url", "")
    body = comment.get("body", "")
    text = f"*New docs comment from {user}*\n{body}\n<{url}|View on GitHub>"

    if SLACK_WEBHOOK_URL:
        try:
            requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=5)
        except Exception as exc:  # pragma: no cover - network failure
            app.logger.error("Slack notification failed: %s", exc)
    return "", 204


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
