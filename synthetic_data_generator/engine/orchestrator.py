"""Orchestrates sequential execution of all runner scripts.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

RUNNERS = [
    ("Spot", ["python", "synthetic_data_generator/engine/spot/run_spot_engine.py"]),
    (
        "Orderbook L3",
        ["python", "synthetic_data_generator/engine/orderbook/run_orderbook_l3.py"],
    ),
    (
        "Orderbook L2",
        ["python", "synthetic_data_generator/engine/orderbook/run_orderbook_l2.py"],
    ),
    (
        "Orderbook L1",
        ["python", "synthetic_data_generator/engine/orderbook/run_orderbook_l1.py"],
    ),
    (
        "Ticks Trades",
        ["python", "synthetic_data_generator/engine/ticks_and_orderflow/run_ticks_trades.py"],
    ),
    (
        "Ticks Orderflow",
        ["python", "synthetic_data_generator/engine/ticks_and_orderflow/run_ticks_orderflow.py"],
    ),
    (
        "Options Chain",
        ["python", "synthetic_data_generator/engine/options/run_options_chain.py"],
    ),
    (
        "Options IV Surface",
        ["python", "synthetic_data_generator/engine/options/run_options_iv_surface.py"],
    ),
    (
        "Options OI",
        ["python", "synthetic_data_generator/engine/options/run_options_oi.py"],
    ),
    (
        "Greeks Primary",
        ["python", "synthetic_data_generator/engine/greeks_and_greekflow/run_greeks_primary_engine.py"],
    ),
    (
        "Greeks Flow",
        ["python", "synthetic_data_generator/engine/greeks_and_greekflow/run_greeks_flow.py"],
    ),
    (
        "Crossasset Correlation",
        ["python", "synthetic_data_generator/engine/crossasset/run_crossasset_correlation.py"],
    ),
    (
        "Crossasset Funding",
        ["python", "synthetic_data_generator/engine/crossasset/run_crossasset_funding.py"],
    ),
]


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


def run_command(name: str, cmd: list[str], cwd: Path) -> int:
    print(f"[{_timestamp()}] START {name}: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[{_timestamp()}] SUCCESS {name}")
    else:
        stderr = result.stderr.strip()
        print(f"[{_timestamp()}] FAILURE {name} (exit code {result.returncode})")
        if stderr:
            print(stderr)
    return result.returncode


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    for name, cmd in RUNNERS:
        exit_code = run_command(name, cmd, repo_root)
        if exit_code != 0:
            print(f"[{_timestamp()}] ORCHESTRATOR FAILURE: {name} failed")
            return exit_code or 1
    print(f"[{_timestamp()}] ORCHESTRATOR SUCCESS: all runners completed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
