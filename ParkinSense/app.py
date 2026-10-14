#!/usr/bin/env python3
"""
ParkinSense — Top-Level Web Server Entry Point
================================================
Launches the Flask Backend API server.

Usage:
    python app.py
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app import app, logger

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5001))
    logger.info("Starting ParkinSense Web Backend API Server on http://0.0.0.0:%d...", port)
    app.run(host="0.0.0.0", port=port, debug=False)
