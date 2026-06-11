"""
Transfer Component Analysis (TCA) baseline.

Implements the classic kernelized TCA with MMD regularization.
Reference formulation:
  (K L K + mu I) W = (K H K) W Lambda
We take the eigenvectors corresponding to the smallest eigenvalues.
"""
from __future__ import annotations

import numpy as np


def _rbf_kernel(X: np.ndarray, gamma: float) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    sq = np.sum(X * X, axis=1, keepdims=True)
    dist2 = sq + sq.T - 2.0 * (X @ X.T)
    return np.exp(-gamma * dist2)


def _linear_kernel(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    return X @ X.T


class TCA:
    def __init__(self, n_components: int = 20, kernel: str = "linear", gamma: float = 1.0, mu: float = 1.0):
        self.n_components = int(n_components)
        self.kernel = kernel
        self.gamma = float(gamma)
        self.mu = float(mu)
        self.W = None
        self.X_fit_ = None

    def _kernel(self, X: np.ndarray) -> np.ndarray:
        if self.kernel == "linear":
            return _linear_kernel(X)
        if self.kernel == "rbf":
            return _rbf_kernel(X, self.gamma)
        raise ValueError(f"Unsupported kernel: {self.kernel}")

    def fit_transform(self, Xs: np.ndarray, Xt: np.ndarray):
        Xs = np.asarray(Xs, dtype=np.float64)
        Xt = np.asarray(Xt, dtype=np.float64)
        n_s = Xs.shape[0]
        n_t = Xt.shape[0]
        X = np.vstack([Xs, Xt])
        n = n_s + n_t

        K = self._kernel(X)

        # MMD matrix L
        L = np.zeros((n, n), dtype=np.float64)
        L[:n_s, :n_s] = 1.0 / (n_s * n_s)
        L[n_s:, n_s:] = 1.0 / (n_t * n_t)
        L[:n_s, n_s:] = -1.0 / (n_s * n_t)
        L[n_s:, :n_s] = -1.0 / (n_s * n_t)

        # Centering matrix H
        H = np.eye(n, dtype=np.float64) - (1.0 / n) * np.ones((n, n), dtype=np.float64)

        A = K @ L @ K + self.mu * np.eye(n, dtype=np.float64)
        B = K @ H @ K

        # Solve generalized eigenproblem via pinv(B) @ A
        eps = 1e-8
        B_reg = B + eps * np.eye(n, dtype=np.float64)
        M = np.linalg.pinv(B_reg) @ A
        eigvals, eigvecs = np.linalg.eigh(M)
        idx = np.argsort(eigvals)[: self.n_components]
        W = eigvecs[:, idx]

        self.W = W
        self.X_fit_ = X

        Z = K @ W
        Zs = Z[:n_s]
        Zt = Z[n_s:]
        return Zs, Zt
