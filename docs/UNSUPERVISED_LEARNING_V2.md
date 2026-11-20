# Unsupervised Learning – Reproducible Pipeline (v2)

Code: `src/unsupervised_pipeline.py` · Notebook: `notebooks/03_unsupervised_reproducible.ipynb`
Outputs: `results/unsupervised_learning/v2_reproducible/`

```bash
pip install -r requirements.txt
python src/unsupervised_pipeline.py          # ~3-5 min on a 2-core CPU
python src/unsupervised_pipeline.py --fast   # quick smoke test
```

## Issues fixed relative to notebook 02

| Issue | Effect | Fix |
|---|---|---|
| `data/hsv_cache/test_features.npy` is not L2-normalised (mean row norm 0.30 vs 1.00 for train/valid) | Test ARI for GMM K=30 dropped to 0.140 (the 0.144 reported in notebook 02) | Default to `test_features_original_pipeline.npy`; automatic row-norm consistency check |
| Validation and test features produced by two different code paths (GPU PCA vs. sklearn re-implementation) | Validation and test scores were not comparable | One `FeaturePipeline` object, fitted on train only, applied to every split |
| K selected using ARI/NMI (label-based) on validation | Model selection was not label-free | K-Means K by silhouette, GMM components by BIC, on train only |
| Result tables hard-coded in notebook cells | Not reproducible | All tables written by the script to CSV/JSON |

## Pipeline

per-block StandardScaler → per-block PCA (HOG 256, HSV 64, LBP 10) → equal block weighting → global PCA 50D (70.1 % variance) → clustering. Seed 42.

## Clustering results (test, 5,273 images, 19 classes)

| Model | ARI | NMI | Purity | Hungarian acc. |
|---|---:|---:|---:|---:|
| K-Means K=12 (silhouette-selected) | 0.178 | 0.398 | 0.301 | 0.294 |
| K-Means K=19 | 0.199 | 0.436 | 0.373 | 0.354 |
| GMM diag K=19 | 0.203 | 0.433 | 0.403 | 0.380 |
| GMM diag K=60 (BIC-selected) | 0.243 | 0.538 | 0.601 | 0.331 |
| Agglomerative Ward K=19 | 0.175 | 0.421 | 0.336 | 0.326 |
| UMAP + K-Means K=19 | **0.291** | **0.556** | 0.460 | **0.414** |
| UMAP + HDBSCAN (25.6 % noise) | 0.167 | 0.531 | 0.501 | 0.334 |

Validation and test scores agree to within ~0.01 for every model, confirming the pipeline generalises.
BIC was still decreasing at 60 components (edge of grid), so BIC favours many small components; purity rises with K by construction, so compare models at equal K using ARI/NMI/Hungarian accuracy.

## Unsupervised disease detection (healthy vs diseased, test)

| Detector | ROC-AUC | Avg. precision |
|---|---:|---:|
| One-class GMM (full cov., trained on healthy leaves only) | **0.912** | 0.982 |
| One-class k-NN distance (healthy only) | 0.852 | 0.965 |
| One-class Isolation Forest (healthy only) | 0.597 | 0.897 |
| Isolation Forest on all training data (no labels) | 0.431 | 0.859 |

Diseased leaves make up 87 % of the data, so a fully label-free outlier detector treats the minority healthy leaves as the anomalies (AUC < 0.5). Training a density model on healthy leaves only (one-class / semi-supervised) works well.

## Observations

- Clusters separate plant species much better than diseases: all groundnut classes form one island in the UMAP map.
- Visually distinctive tomato diseases (Leaf Mold, Late blight, Yellow Leaf Curl Virus, healthy) form nearly pure clusters.
- Handcrafted features limit clustering quality; DINOv2/CNN embeddings (Stage 4) are the natural next input to this same pipeline.
