"""
CORAL (Correlation Alignment) baseline (traditional, closed-form).
"""
from __future__ import annotations

import numpy as np


def _cov(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    X = X - X.mean(axis=0, keepdims=True)
    n = X.shape[0]
    if n <= 1:
        return np.eye(X.shape[1], dtype=np.float64)
    return (X.T @ X) / (n - 1)


def _sqrtm_psd(C: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # symmetric PSD sqrt via eig
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def _invsqrtm_psd(C: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, eps)
    return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T


class CORAL:
    def __init__(self, eps: float = 1e-8):
        self.eps = float(eps)

    def fit_transform(self, Xs: np.ndarray, Xt: np.ndarray):
        Xs = np.asarray(Xs, dtype=np.float64)
        Xt = np.asarray(Xt, dtype=np.float64)

        Cs = _cov(Xs) + self.eps * np.eye(Xs.shape[1], dtype=np.float64)
        Ct = _cov(Xt) + self.eps * np.eye(Xt.shape[1], dtype=np.float64)

        A = _invsqrtm_psd(Cs, self.eps) @ _sqrtm_psd(Ct, self.eps)
        Xs_new = Xs @ A
        Xt_new = Xt
        return Xs_new, Xt_new
