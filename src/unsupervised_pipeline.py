"""
Reproducible unsupervised learning pipeline for Multi-Plant Leaf Disease Detection.

What this script does
---------------------
1. Loads the cached HOG / HSV / LBP features and checks that train, valid and
   test were produced by the same feature pipeline (row-norm consistency check).
2. Builds ONE preprocessing pipeline, fitted on the training split only:
       per-block StandardScaler -> per-block PCA -> equal block weighting
       -> global PCA (50D)
   The exact same fitted object is applied to validation and test, so
   validation and test results are directly comparable.
3. Clustering (labels are NEVER used for fitting or for choosing K):
       - K-Means      (K chosen by silhouette on a training subsample)
       - GMM          (components chosen by BIC on training data)
       - Agglomerative Ward (fitted on a training subsample)
       - UMAP + K-Means / UMAP + HDBSCAN (non-linear manifold clustering)
   K = 19 (the known number of classes) is also reported as a reference.
4. Unsupervised disease detection (anomaly detection):
       - One-class models trained on HEALTHY training leaves only
         (GMM density, k-NN distance, Isolation Forest); diseased leaves
         should look "abnormal".
       - Fully unsupervised Isolation Forest trained on all training leaves.
   Evaluated with ROC-AUC / Average Precision (diseased = positive).
5. Labels are used only AFTER fitting, for external evaluation
   (ARI, NMI, purity, Hungarian accuracy, homogeneity, completeness).

Usage
-----
    python src/unsupervised_pipeline.py                # full run
    python src/unsupervised_pipeline.py --fast         # quicker smoke test
    python src/unsupervised_pipeline.py --test-file test_features.npy
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import HDBSCAN, AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results" / "unsupervised_learning" / "v2_reproducible"

BLOCKS = ("hog", "hsv", "lbp")
# Per-block PCA sizes. Smaller than the original 2044/277/3 so the pipeline is
# fast on a laptop CPU; the global PCA reduces to 50D afterwards anyway.
BLOCK_DIMS = {"hog": 256, "hsv": 64, "lbp": 10}
GLOBAL_DIM = 50
HEALTHY_CLASSES = ("Groundnut_healthy_leaf", "Tomato___healthy", "potato_healthy")


# --------------------------------------------------------------------------
# 1. Data loading and consistency checks
# --------------------------------------------------------------------------
def load_split(split: str, test_file: str) -> tuple[dict[str, np.ndarray], np.ndarray]:
    feats, labels = {}, None
    for b in BLOCKS:
        folder = DATA / f"{b}_cache"
        fname = test_file if split == "test" else f"{split}_features.npy"
        feats[b] = np.load(folder / fname).astype(np.float32)
        y = np.load(folder / f"{split}_labels.npy")
        if labels is None:
            labels = y
        elif not np.array_equal(labels, y):
            raise ValueError(f"Label mismatch between blocks for split '{split}'")
        if len(feats[b]) != len(labels):
            raise ValueError(f"{b}/{split}: features and labels have different lengths")
    return feats, labels


def check_consistency(train: dict, other: dict, name: str, tol: float = 0.15) -> list[str]:
    """Flag feature blocks whose mean row L2-norm differs from train by > tol."""
    issues = []
    for b in BLOCKS:
        n_tr = float(np.linalg.norm(train[b], axis=1).mean())
        n_ot = float(np.linalg.norm(other[b], axis=1).mean())
        rel = abs(n_ot - n_tr) / max(n_tr, 1e-12)
        status = "OK" if rel <= tol else "MISMATCH"
        print(f"  [{status}] {b.upper():3s} {name}: mean row-norm {n_ot:.3f} vs train {n_tr:.3f}")
        if rel > tol:
            issues.append(f"{b}:{name}")
    return issues


# --------------------------------------------------------------------------
# 2. Preprocessing pipeline (fitted on train only)
# --------------------------------------------------------------------------
@dataclass
class FeaturePipeline:
    block_dims: dict = field(default_factory=lambda: dict(BLOCK_DIMS))
    global_dim: int = GLOBAL_DIM
    seed: int = SEED

    def fit(self, feats: dict[str, np.ndarray]) -> "FeaturePipeline":
        self.scalers_, self.pcas_, self.weights_ = {}, {}, {}
        blocks = []
        for b in BLOCKS:
            sc = StandardScaler().fit(feats[b])
            X = sc.transform(feats[b]).astype(np.float32)
            n = min(self.block_dims[b], X.shape[1])
            pca = PCA(n_components=n, svd_solver="randomized" if n < X.shape[1] else "full",
                      random_state=self.seed).fit(X)
            Z = pca.transform(X)
            # Equal contribution: each block gets the same mean squared row norm.
            w = 1.0 / np.sqrt(np.mean(np.sum(Z.astype(np.float64) ** 2, axis=1)))
            self.scalers_[b], self.pcas_[b], self.weights_[b] = sc, pca, float(w)
            blocks.append((Z * w).astype(np.float32))
            print(f"  {b.upper():3s}: {feats[b].shape[1]} -> {n} dims, "
                  f"variance kept {pca.explained_variance_ratio_.sum() * 100:.1f}%")
        Xc = np.hstack(blocks)
        self.global_pca_ = PCA(n_components=self.global_dim, random_state=self.seed).fit(Xc)
        print(f"  Global PCA: {Xc.shape[1]} -> {self.global_dim} dims, variance kept "
              f"{self.global_pca_.explained_variance_ratio_.sum() * 100:.1f}%")
        return self

    def transform(self, feats: dict[str, np.ndarray]) -> np.ndarray:
        blocks = []
        for b in BLOCKS:
            X = self.scalers_[b].transform(feats[b]).astype(np.float32)
            blocks.append((self.pcas_[b].transform(X) * self.weights_[b]).astype(np.float32))
        return self.global_pca_.transform(np.hstack(blocks)).astype(np.float32)


# --------------------------------------------------------------------------
# 3. Evaluation helpers
# --------------------------------------------------------------------------
def purity(y_true, y_pred) -> float:
    cm = contingency_matrix(y_true, y_pred)
    return float(cm.max(axis=0).sum() / cm.sum())


def hungarian_accuracy(y_true, y_pred) -> float:
    """One-to-one cluster->class matching accuracy (unmatched clusters count as errors)."""
    cm = contingency_matrix(y_true, y_pred)
    r, c = linear_sum_assignment(-cm)
    return float(cm[r, c].sum() / cm.sum())


def evaluate(X, y_true, y_pred, model: str, split: str, seed: int = SEED) -> dict:
    mask = y_pred >= 0  # HDBSCAN noise = -1
    yp, yt, Xm = y_pred[mask], y_true[mask], X[mask]
    n_clusters = len(np.unique(yp))
    row = {"Model": model, "Split": split, "Clusters": n_clusters,
           "Noise_%": round(100 * (1 - mask.mean()), 2)}
    if n_clusters > 1:
        idx = np.random.default_rng(seed).choice(len(Xm), size=min(4000, len(Xm)), replace=False)
        row["Silhouette"] = silhouette_score(Xm[idx], yp[idx])
        row["Davies_Bouldin"] = davies_bouldin_score(Xm, yp)
        row["Calinski_Harabasz"] = calinski_harabasz_score(Xm, yp)
    # External metrics on ALL samples (noise treated as its own cluster)
    row.update({
        "ARI": adjusted_rand_score(y_true, y_pred),
        "NMI": normalized_mutual_info_score(y_true, y_pred),
        "Purity": purity(y_true, y_pred),
        "Hungarian_Acc": hungarian_accuracy(y_true, y_pred),
        "Homogeneity": homogeneity_score(y_true, y_pred),
        "Completeness": completeness_score(y_true, y_pred),
    })
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}


# --------------------------------------------------------------------------
# 4. Label-free model selection
# --------------------------------------------------------------------------
def select_k_kmeans(X, k_values, seed=SEED) -> tuple[int, pd.DataFrame]:
    idx = np.random.default_rng(seed).choice(len(X), size=min(6000, len(X)), replace=False)
    rows = []
    for k in k_values:
        km = KMeans(n_clusters=k, n_init=5, random_state=seed).fit(X)
        lab = km.labels_
        rows.append({"K": k, "Inertia": km.inertia_,
                     "Silhouette": silhouette_score(X[idx], lab[idx]),
                     "Davies_Bouldin": davies_bouldin_score(X, lab)})
    df = pd.DataFrame(rows)
    # Ignore trivially small K (<8) which always wins silhouette on overlapping data.
    cand = df[df.K >= 8]
    return int(cand.loc[cand.Silhouette.idxmax(), "K"]), df


def select_k_gmm(X, k_values, cov="diag", seed=SEED) -> tuple[int, pd.DataFrame]:
    rows = []
    for k in k_values:
        g = GaussianMixture(n_components=k, covariance_type=cov, reg_covar=1e-4,
                            max_iter=300, n_init=2, random_state=seed).fit(X)
        rows.append({"K": k, "BIC": g.bic(X), "AIC": g.aic(X), "Converged": g.converged_})
    df = pd.DataFrame(rows)
    return int(df.loc[df.BIC.idxmin(), "K"]), df


# --------------------------------------------------------------------------
# 5. Main experiment
# --------------------------------------------------------------------------
def main(args):
    global OUT
    if args.out:
        OUT = Path(args.out)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    log = {"seed": SEED, "test_file": args.test_file, "block_dims": BLOCK_DIMS,
           "global_dim": GLOBAL_DIM}

    print("1) Loading features")
    tr, y_tr = load_split("train", args.test_file)
    va, y_va = load_split("valid", args.test_file)
    te, y_te = load_split("test", args.test_file)
    print(f"  train={len(y_tr)}  valid={len(y_va)}  test={len(y_te)}  classes={len(np.unique(y_tr))}")
    print("  Feature consistency check (mean L2 row-norm):")
    issues = check_consistency(tr, va, "valid") + check_consistency(tr, te, "test")
    log["consistency_issues"] = issues
    if issues:
        print(f"  WARNING: inconsistent feature extraction detected: {issues}")

    print("\n2) Fitting preprocessing on TRAIN only")
    pipe = FeaturePipeline().fit(tr)
    X = {"train": pipe.transform(tr), "valid": pipe.transform(va), "test": pipe.transform(te)}
    Y = {"train": y_tr, "valid": y_va, "test": y_te}
    del tr, va, te
    log["global_pca_variance_%"] = round(float(pipe.global_pca_.explained_variance_ratio_.sum() * 100), 2)

    print("\n3) Label-free model selection")
    k_grid = [5, 8, 12, 16, 19, 24, 30, 40] if not args.fast else [8, 19, 30]
    k_km, km_sel = select_k_kmeans(X["train"], k_grid)
    km_sel.to_csv(OUT / "kmeans_selection.csv", index=False)
    print(f"  K-Means: K by silhouette = {k_km}")
    g_grid = [10, 19, 30, 40, 50, 60] if not args.fast else [19, 30]
    k_gmm, gmm_sel = select_k_gmm(X["train"], g_grid)
    gmm_sel.to_csv(OUT / "gmm_selection.csv", index=False)
    print(f"  GMM: components by BIC = {k_gmm}")
    log.update({"kmeans_selected_K": k_km, "gmm_selected_K": k_gmm})

    print("\n4) Clustering (fit on train, predict valid/test)")
    preds: dict[str, dict[str, np.ndarray]] = {}

    def record(name, predict_fn):
        preds[name] = {s: predict_fn(X[s], s) for s in ("valid", "test")}
        print(f"  done: {name}")

    for k in sorted({k_km, 19}):
        km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(X["train"])
        record(f"KMeans_K{k}", lambda Z, s, m=km: m.predict(Z))
    for k in sorted({k_gmm, 19}):
        g = GaussianMixture(n_components=k, covariance_type="diag", reg_covar=1e-4,
                            max_iter=500, n_init=3, random_state=SEED).fit(X["train"])
        record(f"GMM_diag_K{k}", lambda Z, s, m=g: m.predict(Z))

    # Agglomerative Ward on a subsample (O(n^2) memory); new points -> nearest centroid.
    sub = np.random.default_rng(SEED).choice(len(X["train"]), size=min(8000, len(X["train"])), replace=False)
    agg = AgglomerativeClustering(n_clusters=19, linkage="ward").fit(X["train"][sub])
    cents = np.vstack([X["train"][sub][agg.labels_ == c].mean(0) for c in range(19)])
    nn_c = NearestNeighbors(n_neighbors=1).fit(cents)
    record("Agglomerative_Ward_K19", lambda Z, s: nn_c.kneighbors(Z, return_distance=False).ravel())

    # Non-linear manifold: UMAP (fit on train) -> K-Means / HDBSCAN
    try:
        import umap
        print("  fitting UMAP (10D) on train ...")
        reducer = umap.UMAP(n_components=10, n_neighbors=30, min_dist=0.0,
                            random_state=SEED, n_jobs=1).fit(X["train"])
        U = {"train": reducer.embedding_,
             "valid": reducer.transform(X["valid"]), "test": reducer.transform(X["test"])}
        for k in sorted({k_km, 19}):
            kmu = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(U["train"])
            preds[f"UMAP_KMeans_K{k}"] = {s: kmu.predict(U[s]) for s in ("valid", "test")}
            print(f"  done: UMAP_KMeans_K{k}")
        hdb = HDBSCAN(min_cluster_size=150, min_samples=10).fit(U["train"])
        keep = hdb.labels_ >= 0
        knn = KNeighborsClassifier(n_neighbors=15).fit(U["train"][keep], hdb.labels_[keep])
        # Points far from any dense region stay noise (-1).
        nn_tr = NearestNeighbors(n_neighbors=15).fit(U["train"][keep])
        d_ref = np.quantile(nn_tr.kneighbors(U["train"][keep])[0].mean(1), 0.95)

        def hdb_predict(Z):
            lab = knn.predict(Z)
            far = nn_tr.kneighbors(Z)[0].mean(1) > d_ref
            lab[far] = -1
            return lab
        preds["UMAP_HDBSCAN"] = {s: hdb_predict(U[s]) for s in ("valid", "test")}
        log["hdbscan_train_clusters"] = int(len(set(hdb.labels_)) - (1 if -1 in hdb.labels_ else 0))
        print(f"  done: UMAP_HDBSCAN ({log['hdbscan_train_clusters']} clusters on train)")
        has_umap = True
    except ImportError:
        print("  umap-learn not installed -> skipping UMAP experiments (pip install umap-learn)")
        has_umap, U = False, None

    rows = []
    for name, d in preds.items():
        for s in ("valid", "test"):
            # Internal metrics always computed in the 50D PCA space for comparability.
            rows.append(evaluate(X[s], Y[s], d[s], name, s))
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "clustering_results.csv", index=False)
    print("\n" + res[res.Split == "test"].drop(columns=["Split"]).to_string(index=False))

    print("\n5) Unsupervised disease detection (healthy vs diseased)")
    is_dis = {s: (~np.isin(Y[s], HEALTHY_CLASSES)).astype(int) for s in Y}
    Xh = X["train"][is_dis["train"] == 0]
    det_rows, scores = [], {}
    gh_k, _ = select_k_gmm(Xh, [2, 4, 8, 12, 16], cov="full")
    gh = GaussianMixture(n_components=gh_k, covariance_type="full", reg_covar=1e-3,
                         n_init=3, random_state=SEED).fit(Xh)
    knn_h = NearestNeighbors(n_neighbors=10).fit(Xh)
    iso_h = IsolationForest(n_estimators=300, random_state=SEED).fit(Xh)
    iso_all = IsolationForest(n_estimators=300, random_state=SEED).fit(X["train"])
    detectors = {
        f"OneClass_GMM_full_K{gh_k} (healthy-only)": lambda Z: -gh.score_samples(Z),
        "OneClass_kNN_distance (healthy-only)": lambda Z: knn_h.kneighbors(Z)[0].mean(1),
        "OneClass_IsolationForest (healthy-only)": lambda Z: -iso_h.score_samples(Z),
        "IsolationForest (all train, no labels)": lambda Z: -iso_all.score_samples(Z),
    }
    for name, fn in detectors.items():
        for s in ("valid", "test"):
            sc = fn(X[s])
            scores[(name, s)] = sc
            det_rows.append({"Detector": name, "Split": s,
                             "ROC_AUC": round(roc_auc_score(is_dis[s], sc), 4),
                             "Avg_Precision": round(average_precision_score(is_dis[s], sc), 4),
                             "Diseased_rate": round(is_dis[s].mean(), 4)})
    det = pd.DataFrame(det_rows)
    det.to_csv(OUT / "disease_detection_results.csv", index=False)
    print(det.to_string(index=False))

    # ----------------------------------------------------------------------
    # 6. Figures
    # ----------------------------------------------------------------------
    print("\n6) Saving figures")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    test_res = res[res.Split == "test"].set_index("Model")
    best = test_res["NMI"].idxmax()
    log["best_model_by_test_NMI"] = best

    # (a) Model comparison
    fig, ax = plt.subplots(figsize=(11, 5))
    test_res[["ARI", "NMI", "Purity", "Hungarian_Acc"]].plot.bar(ax=ax)
    ax.set_title("Clustering vs ground truth on TEST (labels used only for evaluation)")
    ax.set_ylabel("Score"); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.3)
    plt.xticks(rotation=30, ha="right"); plt.tight_layout()
    plt.savefig(OUT / "01_model_comparison_test.png", dpi=140); plt.close()

    # (b) Contingency heatmap for the best model
    yp = preds[best]["test"]
    ct = pd.crosstab(pd.Series(Y["test"], name="True class"), pd.Series(yp, name="Cluster"),
                     normalize="columns")
    order = ct.idxmax(axis=0).map({c: i for i, c in enumerate(ct.index)}).sort_values().index
    ct = ct[order]
    fig, ax = plt.subplots(figsize=(max(10, .38 * ct.shape[1]), 8))
    im = ax.imshow(ct.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(ct.index))); ax.set_yticklabels(ct.index, fontsize=8)
    ax.set_xticks(range(ct.shape[1])); ax.set_xticklabels(ct.columns, fontsize=7)
    ax.set_xlabel("Cluster"); ax.set_title(f"{best}: class composition of each cluster (test)")
    fig.colorbar(im, ax=ax, label="Fraction of cluster"); plt.tight_layout()
    plt.savefig(OUT / "02_best_model_cluster_composition.png", dpi=140); plt.close()

    # (c) ROC curves for disease detection
    fig, ax = plt.subplots(figsize=(6.5, 6))
    for name in detectors:
        fpr, tpr, _ = roc_curve(is_dis["test"], scores[(name, "test")])
        auc = det[(det.Detector == name) & (det.Split == "test")].ROC_AUC.iloc[0]
        ax.plot(fpr, tpr, label=f"{name}  AUC={auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Unsupervised disease detection (test)"); ax.legend(fontsize=7, loc="lower right")
    plt.tight_layout(); plt.savefig(OUT / "03_disease_detection_roc.png", dpi=140); plt.close()

    # (d) 2D UMAP map of the test set
    if has_umap:
        import umap
        sub2 = np.random.default_rng(SEED).choice(len(X["train"]), size=min(10000, len(X["train"])), replace=False)
        r2 = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=SEED,
                       n_jobs=1).fit(X["train"][sub2])
        E = r2.transform(X["test"])
        classes = np.unique(Y["test"]); cmap = plt.get_cmap("tab20")
        fig, axes = plt.subplots(1, 2, figsize=(17, 7.5))
        for i, c in enumerate(classes):
            m = Y["test"] == c
            axes[0].scatter(E[m, 0], E[m, 1], s=3, color=cmap(i % 20), label=c)
        axes[0].set_title("Test images - coloured by TRUE disease class")
        axes[0].legend(markerscale=4, fontsize=6, ncol=2, loc="best")
        axes[1].scatter(E[:, 0], E[:, 1], s=3, c=yp, cmap="tab20")
        axes[1].set_title(f"Test images - coloured by {best} cluster")
        for a in axes: a.set_xticks([]); a.set_yticks([])
        plt.tight_layout(); plt.savefig(OUT / "04_umap_test_true_vs_clusters.png", dpi=130); plt.close()

    log["runtime_minutes"] = round((time.time() - t0) / 60, 2)
    with open(OUT / "summary.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"\nAll outputs saved to {OUT}  ({log['runtime_minutes']} min)")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-file", default="test_features_original_pipeline.npy",
                   help="Test feature file inside each *_cache folder. The default one is "
                        "extracted with the same pipeline as train/valid.")
    p.add_argument("--out", default=None, help="Output folder (default: results/unsupervised_learning/v2_reproducible).")
    p.add_argument("--fast", action="store_true", help="Smaller model-selection grids.")
    main(p.parse_args())
