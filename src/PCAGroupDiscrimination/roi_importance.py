"""
Extract and summarize ROI-level feature importance from saved ML pipelines.

Supported feature sets (set 2 is ignored):
  0 – flattened ROI×time signal (Fortran order: all ROIs at t=0, then t=1, …)
  1 – per-experiment blocks of [std, abs-mean, window-means…] per ROI
      (matches ProcessSignals.extract_sequence_features1)

Importance methods per model:
  LR, SVM  – |coefficients| mapped through SelectKBest
  DTree, RandForest – feature_importances_ mapped through SelectFromModel
  NN       – mean |input gradients| (DeepSklearnWrapper.feature_importances_)
"""

import os
import re
import glob
import warnings

import numpy as np
import pandas as pd


def _fetch_atlas_labels(n_rois=100):
    """Load Schaefer atlas ROI labels (same source as build_signals_from_files)."""
    from nilearn import datasets

    dataset = datasets.fetch_atlas_schaefer_2018(
        n_rois=n_rois, yeo_networks=7, resolution_mm=2
    )
    return [
        label.decode("utf-8") if isinstance(label, bytes) else str(label)
        for label in dataset.labels
    ]


def infer_extra_features_set(base_directory):
    """Infer feature set from parent folder name like .../full_pipeline_set1/2_ml_pc-0."""
    path = os.path.abspath(base_directory)
    for part in reversed(path.split(os.sep)):
        match = re.search(r"full_pipeline_set(\d+)", part, re.IGNORECASE)
        if match:
            return int(match.group(1))
    return 1


class FeatureLayout:
    """Maps flat feature indices to ROI indices for feature sets 0 and 1."""

    FEATURE_SET0 = 0
    FEATURE_SET1 = 1

    BLOCK_NAMES_SET1 = ("std", "abs_mean")  # followed by win_1 … win_N

    def __init__(self, n_regions, extra_features_set, n_features, n_windows_per_exp=6, n_experiments=2):
        self.n_regions = int(n_regions)
        self.extra_features_set = int(extra_features_set)
        self.n_features = int(n_features)
        self.n_windows_per_exp = int(n_windows_per_exp)
        self.n_experiments = int(n_experiments)

        if self.extra_features_set not in (self.FEATURE_SET0, self.FEATURE_SET1):
            raise ValueError(f"Unsupported extra_features_set={extra_features_set} (set 2 ignored).")

        self._roi_index = None
        self._feature_meta = None
        self._build_mapping()

    @classmethod
    def from_n_features(cls, n_features, n_regions, extra_features_set, n_windows_per_exp=6):
        """Construct layout, inferring n_experiments / n_windows when needed."""
        if extra_features_set == cls.FEATURE_SET0:
            if n_features % n_regions != 0:
                raise ValueError(
                    f"Set 0: n_features ({n_features}) must be divisible by n_regions ({n_regions})."
                )
            return cls(
                n_regions=n_regions,
                extra_features_set=extra_features_set,
                n_features=n_features,
                n_windows_per_exp=n_windows_per_exp,
                n_experiments=1,
            )

        # Set 1: n_features = n_experiments * (2 + n_windows) * n_regions
        blocks_per_exp = None
        n_experiments = None

        for n_exp in (2, 1):
            if n_features % (n_exp * n_regions) != 0:
                continue
            candidate_blocks = n_features // (n_exp * n_regions)
            candidate_windows = candidate_blocks - 2
            if candidate_windows >= 1 and candidate_windows == n_windows_per_exp:
                blocks_per_exp = candidate_blocks
                n_experiments = n_exp
                break

        if blocks_per_exp is None:
            for n_exp in (2, 1):
                if n_features % (n_exp * n_regions) != 0:
                    continue
                candidate_blocks = n_features // (n_exp * n_regions)
                candidate_windows = candidate_blocks - 2
                if candidate_windows >= 1:
                    blocks_per_exp = candidate_blocks
                    n_experiments = n_exp
                    n_windows_per_exp = candidate_windows
                    break

        if blocks_per_exp is None:
            raise ValueError(
                f"Set 1: cannot infer layout from n_features={n_features}, n_regions={n_regions}."
            )

        return cls(
            n_regions=n_regions,
            extra_features_set=extra_features_set,
            n_features=n_features,
            n_windows_per_exp=n_windows_per_exp,
            n_experiments=n_experiments,
        )

    def _build_mapping(self):
        n_r = self.n_regions
        roi_index = np.zeros(self.n_features, dtype=int)
        meta = []

        if self.extra_features_set == self.FEATURE_SET0:
            n_timepoints = self.n_features // n_r
            for feat_idx in range(self.n_features):
                roi = feat_idx % n_r
                timepoint = feat_idx // n_r
                roi_index[feat_idx] = roi
                meta.append({"feature_idx": feat_idx, "roi": roi, "timepoint": timepoint, "block": "signal"})

        else:
            blocks_per_exp = 2 + self.n_windows_per_exp
            block_names = list(self.BLOCK_NAMES_SET1) + [
                f"win_{i + 1}" for i in range(self.n_windows_per_exp)
            ]
            for feat_idx in range(self.n_features):
                local = feat_idx % (blocks_per_exp * n_r)
                exp = feat_idx // (blocks_per_exp * n_r)
                block = local // n_r
                roi = local % n_r
                roi_index[feat_idx] = roi
                meta.append(
                    {
                        "feature_idx": feat_idx,
                        "roi": roi,
                        "experiment": exp,
                        "block": block_names[block],
                    }
                )

        self._roi_index = roi_index
        self._feature_meta = meta

    @property
    def roi_index_for_feature(self):
        return self._roi_index

    def aggregate_to_roi(self, feature_importances, selected_mask=None, method="mean_abs"):
        """
        Collapse per-feature importances to per-ROI scores.

        selected_mask – boolean array over the full feature space (True = kept by
                        SelectKBest / SelectFromModel). When provided, aggregation
                        uses only selected features for each ROI.

        method='mean_abs' – mean of |importance| over selected features per ROI
        method='sum_abs'  – sum of |importance| over selected features per ROI
        """
        imp = np.asarray(feature_importances, dtype=float).ravel()
        if imp.shape[0] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} importances, got {imp.shape[0]}."
            )

        if selected_mask is not None:
            selected_mask = np.asarray(selected_mask, dtype=bool).ravel()
            if selected_mask.shape[0] != self.n_features:
                raise ValueError(
                    f"selected_mask length {selected_mask.shape[0]} != n_features {self.n_features}."
                )
        else:
            selected_mask = np.ones(self.n_features, dtype=bool)

        abs_imp = np.abs(imp)
        roi_scores = np.zeros(self.n_regions, dtype=float)
        roi_counts = np.zeros(self.n_regions, dtype=int)

        for feat_idx, roi in enumerate(self._roi_index):
            if not selected_mask[feat_idx]:
                continue
            roi_scores[roi] += abs_imp[feat_idx]
            roi_counts[roi] += 1

        if method == "mean_abs":
            with np.errstate(divide="ignore", invalid="ignore"):
                roi_scores = np.where(roi_counts > 0, roi_scores / roi_counts, 0.0)
        elif method != "sum_abs":
            raise ValueError(f"Unknown aggregation method: {method}")

        return roi_scores


def validate_feature_layout_against_process_signals(n_regions=10, n_timepoints=24, n_windows=6):
    """
    Sanity-check that FeatureLayout matches extract_sequence_features1 ordering
    and set-0 Fortran flatten order (mirrors ProcessSignals logic).
    """
    # --- Set 0 ---
    n_r, n_t = n_regions, n_timepoints
    raw = np.zeros((2, n_r * n_t))
    for roi in range(n_r):
        for t in range(n_t):
            raw[0, roi + t * n_r] = 1000 + roi * 100 + t  # Fortran flatten index

    layout0 = FeatureLayout.from_n_features(
        n_features=raw.shape[1], n_regions=n_r, extra_features_set=0
    )
    for roi in range(n_r):
        mask = layout0.roi_index_for_feature == roi
        vals = raw[0, mask]
        expected = np.array([1000 + roi * 100 + t for t in range(n_t)])
        if not np.allclose(vals, expected):
            raise AssertionError(f"Set 0 ROI {roi} mapping mismatch.")

    # --- Set 1 (inline replica of extract_sequence_features1) ---
    exp1_end = n_timepoints // 2
    raw1 = np.random.RandomState(0).randn(1, n_r * n_timepoints)
    subj = raw1[0].reshape(n_timepoints, n_r)
    exp_blocks = [subj[:exp1_end, :], subj[exp1_end:, :]]

    feat_list = []
    for exp_data in exp_blocks:
        feat_list.extend(np.std(exp_data, axis=0))
        feat_list.extend(np.mean(np.abs(exp_data), axis=0))
        for window_data in np.array_split(exp_data, n_windows, axis=0):
            feat_list.extend(np.mean(window_data, axis=0))
    feats = np.array(feat_list)[None, :]

    layout1 = FeatureLayout.from_n_features(
        n_features=feats.shape[1], n_regions=n_r, extra_features_set=1, n_windows_per_exp=n_windows
    )

    block_names = ["std", "abs_mean"] + [f"win_{i + 1}" for i in range(n_windows)]
    expected_by_block = {}
    for exp_i, exp_data in enumerate(exp_blocks):
        std_v = np.std(exp_data, axis=0)
        abs_m = np.mean(np.abs(exp_data), axis=0)
        win_means = [np.mean(w, axis=0) for w in np.array_split(exp_data, n_windows, axis=0)]
        for b_idx, b_name in enumerate(block_names):
            if b_name == "std":
                vec = std_v
            elif b_name == "abs_mean":
                vec = abs_m
            else:
                win_i = int(b_name.split("_")[1]) - 1
                vec = win_means[win_i]
            for roi in range(n_r):
                expected_by_block[(exp_i, b_name, roi)] = vec[roi]

    for meta in layout1._feature_meta:
        feat_idx = meta["feature_idx"]
        key = (meta["experiment"], meta["block"], meta["roi"])
        if not np.isclose(feats[0, feat_idx], expected_by_block[key]):
            raise AssertionError(
                f"Set 1 mismatch at feat {feat_idx}: key={key}, "
                f"got={feats[0, feat_idx]}, expected={expected_by_block[key]}"
            )

    return True


def unwrap_fitted_pipeline(model):
    """
    Unwrap cross-validation or search wrappers (like GridSearchCV)
    to get the core pipeline.
    Crucially: We do NOT strip away sklearn Pipelines, because
    the outer Pipeline often contains the feature selector.
    """
    extracted = model
    while hasattr(extracted, "best_estimator_"):
        extracted = extracted.best_estimator_
    return extracted


def _find_selector(obj):
    """
    Recursively find a feature selector step (e.g., SelectKBest)
    no matter how deep it is nested inside Pipelines.
    """
    if hasattr(obj, "get_support"):
        return obj

    if hasattr(obj, "named_steps"):
        for step in obj.named_steps.values():
            res = _find_selector(step)
            if res is not None:
                return res

    if hasattr(obj, "estimators_") and len(obj.estimators_) > 0:
        return _find_selector(obj.estimators_[0])

    return None


def _get_single_pipeline_importance(pipeline, model_name):
    """
    Helper function to extract importance from a pipeline or estimator.
    Includes robust fallbacks for n_features_in_ and multi-class coef_ arrays.
    """
    selector = _find_selector(pipeline)

    # Extract the final estimator by drilling down pipelines
    estimator = pipeline
    while hasattr(estimator, "named_steps"):
        estimator = list(estimator.named_steps.values())[-1]

    # Helper to get importances for a single raw estimator
    def _get_estimator_imp(est):
        # Unwrap search CV objects just in case they are hiding deep inside
        while hasattr(est, "best_estimator_"):
            est = est.best_estimator_

        if model_name in ("LR", "SVM"):
            if not hasattr(est, "coef_"):
                raise ValueError(f"{model_name} estimator has no coef_. Found wrapper: {type(est).__name__}")
            coef = np.asarray(est.coef_)
            # Handle multi-class (2D) where shape is (n_classes, n_features)
            if coef.ndim > 1 and coef.shape[0] > 1:
                return np.mean(np.abs(coef), axis=0)
            return np.abs(coef).ravel()
        elif model_name in ("DTree", "RandForest", "NN"):
            if not hasattr(est, "feature_importances_"):
                raise ValueError(
                    f"{model_name} estimator has no feature_importances_. Found wrapper: {type(est).__name__}")
            return np.asarray(est.feature_importances_, dtype=float).ravel()
        else:
            raise ValueError(f"Unsupported model: {model_name}")

    # Determine original pre-selection input size
    n_in = None
    if selector is not None:
        n_in = len(selector.get_support())
    else:
        n_in = getattr(estimator, "n_features_in_", None)
        if n_in is None:
            n_in = getattr(estimator, "input_dim", getattr(estimator, "input_dim_", None))

    # Average bagged estimators if the final step is an ensemble
    if hasattr(estimator, "estimators_") and len(estimator.estimators_) > 0:
        imps = [_get_estimator_imp(base_est) for base_est in estimator.estimators_]
        post_imp = np.mean(imps, axis=0)
    else:
        post_imp = _get_estimator_imp(estimator)

    # Ultimate fallback for n_in (assumes no selector was used)
    if n_in is None:
        n_in = len(post_imp)

    n_in = int(n_in)
    full_imp = np.zeros(n_in, dtype=float)
    selected_mask = np.ones(n_in, dtype=bool)

    # Map the selected features back to their original positions
    if selector is not None:
        selected_mask = np.zeros(n_in, dtype=bool)
        selected_idx = selector.get_support(indices=True)
        if len(selected_idx) != len(post_imp):
            raise ValueError(f"Length mismatch: selector kept {len(selected_idx)} but imp is {len(post_imp)}")
        selected_mask[selected_idx] = True
        full_imp[selected_idx] = post_imp
    else:
        full_imp = post_imp

    return full_imp, selected_mask


def extract_full_feature_importances(inner_pipeline, model_name):
    """
    Map model-level importances back to the original feature space.
    Supports single Pipelines, nested Pipelines, and Pipeline Ensembles.
    """
    # Case 1: BaggingClassifier is wrapping the entire Pipeline
    if hasattr(inner_pipeline, "estimators_") and len(inner_pipeline.estimators_) > 0:
        if hasattr(inner_pipeline.estimators_[0], "named_steps"):
            all_imps, all_masks = [], []
            for est in inner_pipeline.estimators_:
                imp, mask = _get_single_pipeline_importance(est, model_name)
                all_imps.append(imp)
                all_masks.append(mask)
            return np.mean(all_imps, axis=0), np.any(all_masks, axis=0)

    # Case 2: Standard Pipeline (or nested pipelines)
    return _get_single_pipeline_importance(inner_pipeline, model_name)

def importance_method_description(model_name):
    descriptions = {
        "LR": "|logistic regression coefficients| (L2, after StandardScaler + SelectKBest)",
        "SVM": "|linear SVM coefficients| (after StandardScaler + SelectKBest)",
        "DTree": "decision tree feature_importances_ (after SelectFromModel)",
        "RandForest": "random forest feature_importances_ (after SelectFromModel)",
        "NN": "mean |input gradients| from DeepSklearnWrapper (after StandardScaler + SelectKBest)",
    }
    return descriptions.get(model_name, "unknown")


class ROIImportanceAnalyzer:
    """Scan ML output folders, extract ROI importances, and write summaries."""

    MODELS = ("LR", "SVM", "DTree", "RandForest", "NN")

    def __init__(
        self,
        base_directory,
        output_directory,
        pc_str=None,
        extra_features_set=None,
        n_windows_per_exp=6,
        top_n=15,
        atlas_labels=None,
    ):
        self.base_directory = base_directory
        self.output_directory = output_directory
        self.pc_str = pc_str
        self.extra_features_set = (
            extra_features_set
            if extra_features_set is not None
            else infer_extra_features_set(base_directory)
        )
        if self.extra_features_set == 2:
            warnings.warn("extra_features_set=2 is ignored for ROI importance analysis.")
            self.extra_features_set = 1
        self.n_windows_per_exp = n_windows_per_exp
        self.top_n = top_n
        self.atlas_labels = atlas_labels or _fetch_atlas_labels()
        self.n_regions = len(self.atlas_labels)

    def _latest_model_path(self, pair_dir, model):
        if self.pc_str:
            pattern = os.path.join(
                pair_dir, f"best_model_pipeline_{model}_{self.pc_str}_*.pkl"
            )
        else:
            pattern = os.path.join(pair_dir, f"best_model_pipeline_{model}_*.pkl")
        files = glob.glob(pattern)
        return max(files, key=os.path.getmtime) if files else None

    def _parse_pair_dir(self, pair_dir):
        folder = os.path.basename(pair_dir)
        match = re.match(r"ml_(.+?)_vs_(.+)", folder)
        if not match:
            return None, None, None
        g0, g1 = match.groups()
        return f"{g0} vs {g1}", g0, g1

    def analyze_pair_model(self, pair_dir, model):
        pair_name, group_0, group_1 = self._parse_pair_dir(pair_dir)
        if pair_name is None:
            return None

        model_path = self._latest_model_path(pair_dir, model)
        if model_path is None:
            return None

        try:
            import joblib

            # Required for unpickling NN pipelines
            from .nn_wrapper import DeepSklearnWrapper  # noqa: F401

            fitted = joblib.load(model_path)
            inner = unwrap_fitted_pipeline(fitted)
            full_imp, selected_mask = extract_full_feature_importances(inner, model)
            layout = FeatureLayout.from_n_features(
                n_features=len(full_imp),
                n_regions=self.n_regions,
                extra_features_set=self.extra_features_set,
                n_windows_per_exp=self.n_windows_per_exp,
            )
            roi_scores = layout.aggregate_to_roi(
                full_imp, selected_mask=selected_mask, method="mean_abs"
            )
        except Exception as exc:
            # warnings.warn(f"ROI importance failed for {pair_name} / {model}: {exc}")
            # return None
            print(f"!!! CRASH in {pair_name} / {model} !!!")
            raise exc

        rows = []
        order = np.argsort(roi_scores)[::-1]
        for rank, roi_idx in enumerate(order, start=1):
            rows.append(
                {
                    "pair": pair_name,
                    "group_0": group_0,
                    "group_1": group_1,
                    "ml_model": model,
                    "roi_index": int(roi_idx),
                    "roi_label": self.atlas_labels[roi_idx],
                    "importance": float(roi_scores[roi_idx]),
                    "rank": rank,
                    "feature_set": self.extra_features_set,
                    "importance_method": importance_method_description(model),
                    "model_path": model_path,
                }
            )
        return rows

    def run(self):
        pair_dirs = sorted(glob.glob(os.path.join(self.base_directory, "ml_*_vs_*")))
        all_rows = []

        for pair_dir in pair_dirs:
            for model in self.MODELS:
                rows = self.analyze_pair_model(pair_dir, model)
                if rows:
                    all_rows.extend(rows)

        if not all_rows:
            print("ROI importance: no saved models found or all extractions failed.")
            return None

        df = pd.DataFrame(all_rows)
        os.makedirs(self.output_directory, exist_ok=True)

        csv_path = os.path.join(self.output_directory, "roi_importance_all.csv")
        df.to_csv(csv_path, index=False)

        top_df = df[df["rank"] <= self.top_n].copy()
        top_csv = os.path.join(self.output_directory, f"roi_importance_top{self.top_n}.csv")
        top_df.to_csv(top_csv, index=False)

        summary_path = os.path.join(self.output_directory, "roi_importance_summary.txt")
        self._write_text_summary(df, top_df, summary_path)

        print(f"ROI importance saved to: {csv_path}")
        print(f"Top-{self.top_n} ROI summary saved to: {top_csv}")
        print(f"ROI importance text summary saved to: {summary_path}")
        return df

    def _write_text_summary(self, df, top_df, path):
        lines = []
        lines.append("=== ROI Importance Summary ===")
        lines.append(f"Feature set: {self.extra_features_set}")
        if self.extra_features_set == 0:
            lines.append(
                "Features: flattened ROI×time signal (Fortran order). "
                "Selected features mapped back via SelectKBest.get_support(indices=True). "
                "ROI score = mean |coef/importance| over selected timepoints only."
            )
        else:
            lines.append(
                f"Features: per movement [std, abs-mean, {self.n_windows_per_exp} window means] × ROI. "
                "Selected features mapped back via get_support(indices=True). "
                "ROI score = mean |coef/importance| over selected derived features only."
            )
        lines.append("")

        for model in self.MODELS:
            sub = top_df[top_df["ml_model"] == model]
            if sub.empty:
                continue
            lines.append(f"--- {model}: {importance_method_description(model)} ---")
            for pair in sub["pair"].unique():
                pair_sub = sub[sub["pair"] == pair].sort_values("rank")
                top_labels = pair_sub["roi_label"].head(5).tolist()
                lines.append(f"  {pair}: {', '.join(top_labels)}")
            lines.append("")

        # ROIs that appear most often in top-5 across all pair/model combinations
        top5 = df[df["rank"] <= 5]
        if not top5.empty:
            freq = (
                top5.groupby(["roi_label", "roi_index"])
                .size()
                .reset_index(name="top5_count")
                .sort_values("top5_count", ascending=False)
            )
            lines.append("--- Most frequent top-5 ROIs (all pairs & models) ---")
            for _, row in freq.head(15).iterrows():
                lines.append(
                    f"  {row['roi_label']} (idx {row['roi_index']}): "
                    f"in top-5 for {row['top5_count']} pair/model runs"
                )

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def summarize_roi_importance(
    base_directory,
    output_directory,
    pc_str=None,
    extra_features_set=None,
    n_windows_per_exp=6,
    top_n=15,
):
    """Entry point called from stat.py."""
    try:
        validate_feature_layout_against_process_signals()
    except Exception as exc:
        warnings.warn(f"Feature layout self-validation failed: {exc}")
    else:
        print("ROI importance: feature layout validated against ProcessSignals.")

    analyzer = ROIImportanceAnalyzer(
        base_directory=base_directory,
        output_directory=output_directory,
        pc_str=pc_str,
        extra_features_set=extra_features_set,
        n_windows_per_exp=n_windows_per_exp,
        top_n=top_n,
    )
    return analyzer.run()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m PCAGroupDiscrimination.roi_importance <2_ml_dir> [output_dir] [pc_str]")
        sys.exit(1)

    base = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(base, "global_summary")
    pc = sys.argv[3] if len(sys.argv) > 3 else None
    validate_feature_layout_against_process_signals()
    print("Feature layout validation passed.")
    summarize_roi_importance(base, out, pc_str=pc)
