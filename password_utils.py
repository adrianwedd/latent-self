"""Utility helpers for password hashing."""

from werkzeug.security import generate_password_hash


def hash_password(password: str) -> str:
    """Return a salted hash suitable for storing in config."""
    return generate_password_hash(password)
