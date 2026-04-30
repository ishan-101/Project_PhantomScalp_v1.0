#!/usr/bin/env python3
"""
Orderbook Orchestrator

Location: synthetic_data_generator/engine/orderbook/orderbook_orchestrator.py

Purpose:
- Run the L3 runner and then the L2 runner from one command.
- Designed to be executed from PowerShell / Terminal as:

    python synthetic_data_generator/engine/orderbook/orderbook_orchestrator.py

Behavior:
1. Resolves canonical project root relative to this file.
2. Locates the two runner scripts:
   - run_orderbook_l3.py
   - run_orderbook_l2.py
3. Executes L3 runner first, then L2 runner (L2 depends on L3 parquet).
4. Streams stdout/stderr to console, raises on failure.
5. Returns non-zero exit code on failure so CI / tooling can detect it.

Notes:
- Uses the current Python interpreter (sys.executable) so it runs inside your venv.
- Runs with working directory = project root to preserve package import semantics.
- No CLI args required; it assumes your runners are zero-argument scripts as discussed.

Save this file to:
    synthetic_data_generator/engine/orderbook/orderbook_orchestrator.py
then run:
    python synthetic_data_generator/engine/orderbook/orderbook_orchestrator.py
"""

from __future__ import annotations
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple


RUNNER_FILENAME_L3 = "run_orderbook_l3.py"
RUNNER_FILENAME_L2 = "run_orderbook_l2.py"


def resolve_paths() -> Tuple[Path, Path, Path]:
    """
    Return (project_root, runner_l3_path, runner_l2_path)
    Assumes this file lives at:
      <project_root>/synthetic_data_generator/engine/orderbook/orderbook_orchestrator.py
    """
    this_file = Path(__file__).resolve()
    orderbook_dir = this_file.parent
    engine_dir = orderbook_dir.parent
    sdg_pkg_dir = engine_dir.parent  # synthetic_data_generator
    project_root = sdg_pkg_dir.parent
    runner_l3 = orderbook_dir / RUNNER_FILENAME_L3
    runner_l2 = orderbook_dir / RUNNER_FILENAME_L2
    return project_root, runner_l3, runner_l2


def check_runner_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Runner not found: {path} (expected).")


def run_script(script_path: Path, cwd: Path) -> int:
    """
    Run the given script with the same Python interpreter and stream output.
    Returns the subprocess returncode.
    """
    print(f"\n--- Running: {script_path.name} ---")
    print(f"Working dir: {cwd}")
    cmd = [sys.executable, str(script_path)]
    start = time.perf_counter()
    try:
        # Using check=True will raise CalledProcessError on non-zero exit code.
        completed = subprocess.run(cmd, cwd=str(cwd), check=True)
        elapsed = time.perf_counter() - start
        print(f"--- {script_path.name} finished OK (elapsed {elapsed:.2f}s) ---\n")
        return 0
    except subprocess.CalledProcessError as e:
        elapsed = time.perf_counter() - start
        print(f"!!! {script_path.name} failed with returncode={e.returncode} (elapsed {elapsed:.2f}s) !!!\n")
        return e.returncode
    except Exception as e:
        elapsed = time.perf_counter() - start
        print(f"!!! {script_path.name} failed with exception: {e} (elapsed {elapsed:.2f}s) !!!\n")
        return 255


def main() -> int:
    project_root, runner_l3, runner_l2 = resolve_paths()
    print(f"Project root resolved to: {project_root}")
    print(f"Orderbook runners resolved to:\n  L3 => {runner_l3}\n  L2 => {runner_l2}")

    # sanity checks
    try:
        check_runner_exists(runner_l3)
    except FileNotFoundError:
        print(f"[ERROR] L3 runner not found at {runner_l3}. Aborting.")
        return 2

    try:
        check_runner_exists(runner_l2)
    except FileNotFoundError:
        print(f"[ERROR] L2 runner not found at {runner_l2}. Aborting.")
        return 3

    # Run L3 first (produces orderbook_l3.parquet)
    rc = run_script(runner_l3, cwd=project_root)
    if rc != 0:
        print("[ERROR] L3 runner failed. Not running L2.")
        return rc

    # Small pause to ensure filesystem has flushed (helpful on some platforms)
    time.sleep(0.2)

    # Run L2 next (consumes orderbook_l3.parquet and produces orderbook_l2.parquet)
    rc = run_script(runner_l2, cwd=project_root)
    if rc != 0:
        print("[ERROR] L2 runner failed.")
        return rc

    print("All orderbook runners completed successfully.")
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
