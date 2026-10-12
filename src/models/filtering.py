from __future__ import annotations

import json
import string
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import minimize

from .reading_order import BoundingBox, DetectedRegion
from .clustering import TextCluster

FEATURE_NAMES: List[str] = [
    "norm_xmin",
    "norm_ymin",
    "norm_xmax",
    "norm_ymax",
    "norm_width",
    "norm_height",
    "norm_area",
    "aspect_ratio",
    "center_x",
    "center_y",
    "margin_dist_min",
    "char_count",
    "word_count",
    "digit_ratio",
    "alpha_ratio",
    "upper_ratio",
    "non_ascii_ratio",
    "punct_ratio",
    "confidence",
    "num_lines",
]


def extract_cluster_features(
    cluster: Union[TextCluster, DetectedRegion],
    page_width: float,
    page_height: float,
) -> List[float]:
    """Extracts a normalized geometric and lexical feature vector from a text cluster or region."""
    pw = max(float(page_width), 1.0)
    ph = max(float(page_height), 1.0)

    bbox = cluster.bbox
    w = max(float(bbox.width), 0.0)
    h = max(float(bbox.height), 0.0)

    n_xmin = float(bbox.xmin) / pw
    n_ymin = float(bbox.ymin) / ph
    n_xmax = float(bbox.xmax) / pw
    n_ymax = float(bbox.ymax) / ph

    n_width = w / pw
    n_height = h / ph
    n_area = (w * h) / (pw * ph)
    aspect_ratio = w / max(h, 1.0)

    center_x = float(bbox.center_x) / pw
    center_y = float(bbox.center_y) / ph

    margin_min = max(0.0, min(n_xmin, 1.0 - n_xmax, n_ymin, 1.0 - n_ymax))

    text = cluster.text or ""
    n_chars = max(len(text), 1)

    char_count = float(len(text))
    word_count = float(len(text.split()))

    digit_ratio = sum(c.isdigit() for c in text) / n_chars
    alpha_ratio = sum(c.isascii() and c.isalpha() for c in text) / n_chars
    upper_ratio = sum(c.isupper() for c in text) / n_chars
    non_ascii_ratio = sum(ord(c) > 127 for c in text) / n_chars
    punct_ratio = sum(c in string.punctuation for c in text) / n_chars

    confidence = float(getattr(cluster, "confidence", 1.0))
    if hasattr(cluster, "regions") and getattr(cluster, "regions") is not None:
        num_lines = float(len(cluster.regions))
    else:
        num_lines = 1.0

    return [
        n_xmin,
        n_ymin,
        n_xmax,
        n_ymax,
        n_width,
        n_height,
        n_area,
        aspect_ratio,
        center_x,
        center_y,
        margin_min,
        char_count,
        word_count,
        digit_ratio,
        alpha_ratio,
        upper_ratio,
        non_ascii_ratio,
        punct_ratio,
        confidence,
        num_lines,
    ]


@dataclass
class FilterModelConfig:
    threshold: float = 0.35
    l2_reg: float = 0.01
    weights_path: Optional[str] = None


class LearnedClusterClassifier:
    """L2-regularized logistic regression classifier for distinguishing dialogue from non-dialogue noise."""

    def __init__(
        self,
        l2_reg: float = 0.01,
        decision_threshold: float = 0.35,
    ) -> None:
        self.l2_reg = l2_reg
        self.decision_threshold = decision_threshold
        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.feature_names: List[str] = list(FEATURE_NAMES)
        self.is_fitted: bool = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> LearnedClusterClassifier:
        """Fits weights using L2-regularized binary cross-entropy minimization."""
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)

        self.mean = np.mean(X_arr, axis=0)
        self.std = np.std(X_arr, axis=0)
        self.std[self.std < 1e-6] = 1.0

        X_norm = (X_arr - self.mean) / self.std
        n_samples, n_features = X_norm.shape

        def loss_and_grad(theta: np.ndarray) -> Tuple[float, np.ndarray]:
            w = theta[:-1]
            b = theta[-1]
            logits = np.dot(X_norm, w) + b
            # Numerically stable binary cross-entropy with L2 penalty
            loss = np.mean(np.logaddexp(0.0, logits) - y_arr * logits) + 0.5 * self.l2_reg * np.sum(w ** 2)
            probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -25.0, 25.0)))
            grad_w = np.mean((probs - y_arr)[:, None] * X_norm, axis=0) + self.l2_reg * w
            grad_b = np.mean(probs - y_arr)
            return float(loss), np.append(grad_w, grad_b)

        init_theta = np.zeros(n_features + 1, dtype=float)
        res = minimize(loss_and_grad, init_theta, jac=True, method="L-BFGS-B")

        self.weights = res.x[:-1]
        self.bias = float(res.x[-1])
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Computes dialogue probabilities for a batch of feature vectors."""
        if not self.is_fitted or self.weights is None or self.mean is None or self.std is None:
            # Safe default fallback: keep all candidates
            return np.ones(len(X), dtype=float)

        X_arr = np.asarray(X, dtype=float)
        X_norm = (X_arr - self.mean) / self.std
        logits = np.dot(X_norm, self.weights) + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -25.0, 25.0)))

    def predict_single(self, features: Union[List[float], np.ndarray]) -> float:
        """Computes dialogue probability for a single feature vector."""
        if not self.is_fitted:
            return 1.0
        X_arr = np.array([features], dtype=float)
        probs = self.predict_proba(X_arr)
        return float(probs[0])

    def save_to_json(self, path: Union[Path, str]) -> None:
        """Exports model parameters, scalers, and metadata to JSON."""
        if not self.is_fitted or self.weights is None:
            raise RuntimeError("Cannot save unfitted classifier.")

        data = {
            "model_type": "LearnedClusterClassifier",
            "feature_names": self.feature_names,
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "mean": self.mean.tolist() if self.mean is not None else [],
            "std": self.std.tolist() if self.std is not None else [],
            "l2_reg": self.l2_reg,
            "decision_threshold": self.decision_threshold,
        }
        target_path = Path(path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, path: Union[Path, str]) -> LearnedClusterClassifier:
        """Loads model parameters from JSON."""
        source_path = Path(path).resolve()
        with open(source_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        clf = cls(
            l2_reg=data.get("l2_reg", 0.01),
            decision_threshold=data.get("decision_threshold", 0.35),
        )
        clf.feature_names = data.get("feature_names", list(FEATURE_NAMES))
        clf.weights = np.array(data["weights"], dtype=float)
        clf.bias = float(data["bias"])
        clf.mean = np.array(data["mean"], dtype=float)
        clf.std = np.array(data["std"], dtype=float)
        clf.is_fitted = True
        return clf


class LearnedNarrationClassifier(LearnedClusterClassifier):
    """Classifier predicting whether dialogue is a NARRATION caption box rather than character speech."""

    def __init__(
        self,
        l2_reg: float = 0.01,
        decision_threshold: float = 0.50,
        threshold: Optional[float] = None,
    ) -> None:
        thresh = threshold if threshold is not None else decision_threshold
        super().__init__(l2_reg=l2_reg, decision_threshold=thresh)

