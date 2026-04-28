from dataclasses import dataclass
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

@dataclass
class ModelArtifact:
    path: str
    auc: float


def train_model(df: pd.DataFrame, cfg) -> ModelArtifact:
    X = df[cfg.ml.features].values
    y = (df[cfg.ml.target] > 0).astype(int).values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    auc = float(roc_auc_score(yte, clf.predict_proba(Xte)[:,1]))

    initial_type = [("input", FloatTensorType([None, X.shape[1]]))]
    onnx_model = convert_sklearn(clf, initial_types=initial_type)
    onnx_path = cfg.ml.onnx_path
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())
    return ModelArtifact(str(onnx_path), auc)