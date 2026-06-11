"""
Subspace Alignment (SA) baseline.

Uses PCA to learn source/target subspaces and aligns source to target.
"""
from __future__ import annotations

import numpy as np


class SA:
    def __init__(self, n_components: int = 20, whiten: bool = False, random_state: int = 0):
        self.n_components = int(n_components)
        self.whiten = bool(whiten)
        self.random_state = int(random_state)
        self.Ps = None
        self.Pt = None

    def fit_transform(self, Xs: np.ndarray, Xt: np.ndarray):
        try:
            from sklearn.decomposition import PCA
        except Exception as exc:
            raise RuntimeError("scikit-learn is required for SA. Run: pip install scikit-learn") from exc

        Xs = np.asarray(Xs, dtype=np.float64)
        Xt = np.asarray(Xt, dtype=np.float64)

        pca_s = PCA(n_components=self.n_components, whiten=self.whiten, random_state=self.random_state)
        pca_t = PCA(n_components=self.n_components, whiten=self.whiten, random_state=self.random_state)

        Ps = pca_s.fit(Xs).components_.T  # (dim, d)
        Pt = pca_t.fit(Xt).components_.T  # (dim, d)

        M = Ps.T @ Pt
        Xs_new = Xs @ Ps @ M
        Xt_new = Xt @ Pt

        self.Ps = Ps
        self.Pt = Pt
        return Xs_new, Xt_new
