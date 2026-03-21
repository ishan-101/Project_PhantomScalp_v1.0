# app/cli_backtest.py
import argparse
from types import SimpleNamespace
from app.orchestrator.controller import run_backtest_v02

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="./out/v02", help="Report output directory")
    args = p.parse_args()
    cfg = SimpleNamespace(output_dir=args.output_dir)  # stand-in for your real config
    result = run_backtest_v02(cfg)
    print("Artifacts:")
    for k, v in result["paths"].items():
        print(f"  {k}: {v}")
    print("KPIs:", result["kpis"])

if __name__ == "__main__":
    main()
