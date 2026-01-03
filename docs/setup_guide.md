# Setup Guide

Follow these steps to get a development environment running quickly.

## Context/Why

You need a clear path to try the project yourself without digging through old docs.

1. Clone the repository.
   ```bash
   git clone https://github.com/your-username/latent-self.git
   cd latent-self
   ```
2. Create a Python 3.11 virtual environment and activate it.
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies from the lock file.
   ```bash
   pip install -r requirements.lock
   ```
4. Run the application.
   ```bash
   python latent_self.py
   ```
