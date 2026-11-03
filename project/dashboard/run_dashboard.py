"""
dashboard/run_dashboard.py
==========================
Launcher script for the AI CPU-GPU Cloud HPC Dashboard.

Usage:
    python dashboard/run_dashboard.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.app import app

if __name__ == "__main__":
    print("=" * 80)
    print("AI-DRIVEN ADAPTIVE CPU-GPU WORKLOAD SCHEDULER")
    print("INTERACTIVE LIVE MONITORING & OPTIMIZATION DASHBOARD")
    print("=" * 80)
    print("Serving live on: http://127.0.0.1:8050")
    print("Press Ctrl+C to terminate server.")
    print("=" * 80)
    app.run(host="127.0.0.1", port=8050, debug=False)

