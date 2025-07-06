#!/usr/bin/env python3
"""Utility to generate a WerkZeug password hash for the admin panel."""
from __future__ import annotations

import argparse
import getpass
from werkzeug.security import generate_password_hash


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a password hash for `admin_password_hash`"
    )
    parser.add_argument(
        "password",
        nargs="?",
        help="Plaintext password. If omitted you will be prompted",
    )
    args = parser.parse_args()

    password = args.password or getpass.getpass("Password: ")
    print(generate_password_hash(password))


if __name__ == "__main__":
    main()
