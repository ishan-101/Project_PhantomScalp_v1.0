"""
app/ml/preprocess.py

Preprocessing pipeline for Project_PhantomScalp v0.2 -> v0.3

Responsibilities:
 - Accept a DataFrame (train / val / test)
 - Select features specified by the user
 - Handle missing values with SimpleImputer
 - Scale numeric features with StandardScaler (optionally MinMaxScaler)
 - Optionally apply PCA (disabled by default)
 - Save / load pipeline object to/from disk using joblib
 - Provide simple CLI to fit & persist a pipeline, and to transform new CSVs

Outputs:
 - artifact: artifacts/scaler_pipeline.pkl

Usage (examples):
  # Fit and save pipeline from train.csv
  python -m app.ml.preprocess fit --train-csv data/train.csv --out artifacts/

  # Transform a CSV using saved pipeline (inference)
  python -m app.ml.preprocess transform --in-csv data/val.csv --pipeline artifacts/scaler_pipeline.pkl --out-csv data/val_scaled.csv
"""

from __future__ import annotations
import argparse
import json
import os
import sys
import logging
from typing import List, Optional, Sequence, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("preprocess")

PIPELINE_FILENAME = "scaler_pipeline.pkl"


def _default_numeric_selector(df: pd.DataFrame) -> List[str]:
    # exclude label columns and non-informative id cols
    exclude = {"index", "idx", "id", "timestamp", "ts", "datetime", "date", "symbol"}
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c.lower() not in exclude]
    return num_cols


def build_preprocessor(
    numeric_features: Sequence[str],
    impute_strategy: str = "median",
    scaler: str = "standard",
    use_pca: bool = False,
    pca_n_components: int | float | None = None,
) -> Pipeline:
    """
    Build a sklearn Pipeline which imputes + scales numeric features and optionally applies PCA.

    Args:
        numeric_features: list of column names to include in the numeric transformer
        impute_strategy: 'mean'|'median'|'most_frequent'|'constant'
        scaler: 'standard'|'minmax'
        use_pca: whether to append PCA
        pca_n_components: if float in (0,1) interprets as variance ratio, if int interprets as n_components
    Returns:
        pip: sklearn Pipeline object
    """
    if scaler not in ("standard", "minmax"):
        raise ValueError("scaler must be 'standard' or 'minmax'")

    num_transformers = []
    # imputer then scaler
    imputer = SimpleImputer(strategy=impute_strategy)
    if scaler == "standard":
        scaler_obj = StandardScaler()
    else:
        scaler_obj = MinMaxScaler()

    numeric_pipeline = Pipeline([("imputer", imputer), ("scaler", scaler_obj)])
    transformers = [("num", numeric_pipeline, list(numeric_features))]

    column_transformer = ColumnTransformer(transformers, remainder="drop", verbose=False)

    steps = [("column_transformer", column_transformer)]
    if use_pca:
        if pca_n_components is None:
            # default: keep 0.99 variance
            pca_n_components = 0.99
        steps.append(("pca", PCA(n_components=pca_n_components)))

    pip = Pipeline(steps)
    return pip


def fit_and_save_pipeline(
    train_df: pd.DataFrame,
    feature_columns: Optional[Sequence[str]],
    out_dir: str,
    impute_strategy: str = "median",
    scaler: str = "standard",
    use_pca: bool = False,
    pca_n_components: int | float | None = None,
) -> str:
    out_dir = out_dir or "artifacts"
    os.makedirs(out_dir, exist_ok=True)
    # decide feature columns
    if not feature_columns:
        feature_columns = _default_numeric_selector(train_df)
        logger.info("Auto-selected numeric features (%d): %s", len(feature_columns), feature_columns[:10])
    else:
        # validate presence
        missing = [c for c in feature_columns if c not in train_df.columns]
        if missing:
            raise ValueError(f"Requested feature columns not found in train_df: {missing}")

    pip = build_preprocessor(feature_columns, impute_strategy=impute_strategy, scaler=scaler, use_pca=use_pca, pca_n_components=pca_n_components)
    logger.info("Fitting preprocessing pipeline on training data (rows=%d, features=%d)", len(train_df), len(feature_columns))
    pip.fit(train_df[feature_columns])
    out_path = os.path.join(out_dir, PIPELINE_FILENAME)
    joblib.dump({"pipeline": pip, "feature_columns": list(feature_columns)}, out_path)
    logger.info("Saved pipeline to %s", out_path)
    return out_path


def load_pipeline(path: str) -> Dict[str, object]:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    data = joblib.load(path)
    if not isinstance(data, dict) or "pipeline" not in data or "feature_columns" not in data:
        raise ValueError("Pipeline file must be a dict with keys ['pipeline','feature_columns']")
    return data


def transform_df_with_pipeline(
    df: pd.DataFrame,
    pipeline_bundle: Dict[str, object],
) -> pd.DataFrame:
    """
    Transform a DataFrame and return a new DataFrame with transformed features.
    The returned DataFrame has columns in same order as feature_columns saved in pipeline bundle.
    If the pipeline includes PCA, column names will be 'PC_0', 'PC_1', ...
    """
    pip = pipeline_bundle["pipeline"]
    feature_columns: List[str] = list(pipeline_bundle["feature_columns"])
    # validate
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Input DataFrame missing required feature columns: {missing}")
    X_trans = pip.transform(df[feature_columns])
    # build column names
    if any(step[0] == "pca" for step in pip.steps):
        # pip.named_steps['pca'] exists
        n_comp = X_trans.shape[1]
        colnames = [f"PC_{i}" for i in range(n_comp)]
    else:
        colnames = feature_columns
    out_df = pd.DataFrame(X_trans, index=df.index, columns=colnames)
    return out_df


# --- CLI utilities ---
def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # preserve index if present
    return pd.read_csv(path, low_memory=False)


def _write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Wrote CSV: %s", path)


def cli_fit(args: argparse.Namespace) -> None:
    train_df = _read_csv(args.train_csv)
    feature_columns = None
    if args.feature_cols_json:
        with open(args.feature_cols_json, "r", encoding="utf8") as fh:
            feature_columns = json.load(fh)
    pipeline_path = fit_and_save_pipeline(
        train_df,
        feature_columns,
        args.out_dir,
        impute_strategy=args.impute_strategy,
        scaler=args.scaler,
        use_pca=args.use_pca,
        pca_n_components=args.pca_n_components,
    )
    print(pipeline_path)


def cli_transform(args: argparse.Namespace) -> None:
    bundle = load_pipeline(args.pipeline)
    df = _read_csv(args.in_csv)
    out_df = transform_df_with_pipeline(df, bundle)
    # optionally join other columns
    if args.join_columns:
        # keep only those columns which exist in in_df
        in_df = pd.read_csv(args.in_csv, low_memory=False)
        join_cols = [c for c in args.join_columns.split(",") if c in in_df.columns]
        if join_cols:
            out_df = pd.concat([in_df[join_cols].reset_index(drop=True), out_df.reset_index(drop=True)], axis=1)
    _write_csv(out_df, args.out_csv)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="app.ml.preprocess", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    fit_p = sub.add_parser("fit", help="Fit preprocessing pipeline on training CSV and save artifacts")
    fit_p.add_argument("--train-csv", required=True)
    fit_p.add_argument("--out-dir", default="artifacts")
    fit_p.add_argument("--feature-cols-json", help="JSON file with list of feature columns to use", dest="feature_cols_json", default=None)
    fit_p.add_argument("--impute-strategy", default="median", choices=["mean", "median", "most_frequent", "constant"])
    fit_p.add_argument("--scaler", default="standard", choices=["standard", "minmax"])
    fit_p.add_argument("--use-pca", action="store_true")
    fit_p.add_argument("--pca-n-components", type=float, default=None)
    transform_p = sub.add_parser("transform", help="Transform a CSV using saved pipeline")
    transform_p.add_argument("--in-csv", required=True)
    transform_p.add_argument("--pipeline", required=True)
    transform_p.add_argument("--out-csv", required=True)
    transform_p.add_argument("--join-columns", default="", help="Comma-separated columns from input to include in output (e.g. timestamp)")
    return p


def main(argv: List[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.cmd == "fit":
        cli_fit(args)
    elif args.cmd == "transform":
        cli_transform(args)
    else:
        raise RuntimeError("unknown command")


if __name__ == "__main__":
    main()
