import os
import json
import time
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import umap

from datetime import datetime
from joblib import Parallel, delayed

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, f_classif, SelectFromModel
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings("ignore", message=".*sklearn.utils.parallel.delayed.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")

from .nn_wrapper import DeepSklearnWrapper
from .utils import now_str, savefig, setup_logger

class MLAnalyzer:
    def __init__(self, args):
        self.args = args
        self.step_prev_folder = os.path.join(self.args.output_dir, '1_clustering_stability')
        self.step_folder = os.path.join(self.args.output_dir, f"2_ml_{args.pc_str}")
        os.makedirs(self.step_folder, exist_ok=True)
        self.logger = setup_logger(self.step_folder)

    def calculate_binary_class(self, row, zero_class=None, one_class=None):
        ins_group = row["instrument_group"]
        is_train_set = row["is_train_set"]
        if not is_train_set:
            return np.nan
        if (zero_class == "NM" and one_class == "mus") or (
            zero_class == "mus" and one_class == "NM"):
            classification = 0 if ins_group == "NM" else 1
        else:
            classification = 0 if ins_group == zero_class else 1 if ins_group == one_class else np.nan
        return classification

    def prepare_data(self, ids, zero_class, one_class):
        """Helper to prepare data for both regular ML and ML stability."""
        feat = pd.read_csv(self.args.metadata_csv)
        unnamed_cols = [c for c in feat.columns if str(c).startswith("Unnamed:")]
        if unnamed_cols:
            feat = feat.drop(columns=unnamed_cols)

        for c in ["has_classical", "has_jazz", "has_rock", "has_pop"]:
            feat[c] = pd.to_numeric(feat[c], errors="coerce").fillna(0).astype(int)

        feat["instrument_group"] = feat["instrument_group"].fillna("NM")
        feat["group"] = feat["group"].astype(str).str.strip()

        feat["y_classical_tendency"] = np.nan
        feat["y_classical_tendency"] = feat.apply(
            self.calculate_binary_class,
            axis=1,
            args=(zero_class, one_class)
        )
        all_metadata_ids = feat["sub_id"].unique()
        missing_ids = set(all_metadata_ids) - set(ids)
        if missing_ids:
            self.logger.info(
                f"[Warning] The following {len(missing_ids)} subjects from metadata were NOT found in processed data:")
            self.logger.info(missing_ids)

        Xmeta = pd.DataFrame({
            "sub_id": ids,
            "x_row_idx": np.arange(len(ids))
        })

        merged = Xmeta.merge(
            feat[["sub_id", "group", "instrument_group", "has_jazz", "has_rock", "has_pop", "y_classical_tendency"]],
            on="sub_id", how="inner"
        )

        if merged.empty:
            raise RuntimeError("No overlap between built PC0 matrix and features CSV")

        return merged[merged["y_classical_tendency"].notnull()]

    def plot_model_manifolds(self, X_proj, oof_preds, data, model_name="ML_Model"):
        """
        Visualizes the feature space used by the model.
        Receives PRE-COMPUTED representations and Out-Of-Fold (OOF) predictions.

        Parameters:
        - X_proj: The space to project.
                  For Deep Models: Pass the extracted Latent Space (e.g., 16D).
                  For Classic Models (SVM, RF): Pass the Raw Feature Matrix (X).
        - oof_preds: The predictions generated during your Cross-Validation phase.
        - data: The metadata dataframe corresponding to the rows in X_proj.
        - model_name: String for the plot titles and file names.

        Produces interactive HTML files (2D + 3D) for PCA, UMAP, and t-SNE.
        """
        logger = setup_logger(self.step_folder)

        # Verify dimensions match before proceeding
        if len(X_proj) != len(data) or len(oof_preds) != len(data):
            logger.error("Dimension mismatch between features, predictions, and metadata. Aborting plot.")
            return

        # 1. Build plot dataframe
        plot_rows = []
        for i, (_, row) in enumerate(data.iterrows()):
            label = row.get("instrument_group", row.get("Group", "Unknown"))

            plot_rows.append({
                'sub_id': row['sub_id'],
                'Group': label,
                'pred_class': str(oof_preds[i])  # Cast to string for categorical coloring/symbols in plotly
            })

        plot_df = pd.DataFrame(plot_rows)
        tag = now_str()

        # Helper to safely create and save plots
        def save_interactive_plot(fig_obj, filename_prefix):
            out_path = os.path.join(self.step_folder_groups, f"{filename_prefix}_{model_name}_{tag}.html")
            fig_obj.write_html(out_path)

        # UMAP 2D
        try:
            reducer2 = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
            X_umap2 = reducer2.fit_transform(X_proj)
            plot_df['UMAP_2D_1'] = X_umap2[:, 0]
            plot_df['UMAP_2D_2'] = X_umap2[:, 1]

            fig_umap2 = px.scatter(plot_df, x='UMAP_2D_1', y='UMAP_2D_2', color='Group', symbol='pred_class',
                                   hover_name='sub_id', title=f'Manifold UMAP 2D ({model_name})')
            save_interactive_plot(fig_umap2, "manifold_umap2d")

            # Save coordinates
            coords_df = plot_df[['sub_id', 'Group', 'pred_class', 'UMAP_2D_1', 'UMAP_2D_2']]
            coords_df.to_csv(os.path.join(self.step_folder_groups, f"umap2d_coordinates_{model_name}_{tag}.csv"),
                             index=False)
        except Exception as e:
            logger.warning(f"UMAP 2D failed: {e}")

        # UMAP 3D
        try:
            reducer3 = umap.UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
            X_umap3 = reducer3.fit_transform(X_proj)
            plot_df['UMAP_3D_1'] = X_umap3[:, 0]
            plot_df['UMAP_3D_2'] = X_umap3[:, 1]
            plot_df['UMAP_3D_3'] = X_umap3[:, 2]

            fig_umap3 = px.scatter_3d(plot_df, x='UMAP_3D_1', y='UMAP_3D_2', z='UMAP_3D_3', color='Group',
                                      symbol='pred_class', hover_name='sub_id',
                                      title=f'Manifold UMAP 3D ({model_name})')
            save_interactive_plot(fig_umap3, "manifold_umap3d")
        except Exception as e:
            logger.warning(f"UMAP 3D failed: {e}")

        # t-SNE 2D / 3D
        try:
            perp = min(30, max(5, len(X_proj) // 3))

            tsne2 = TSNE(n_components=2, perplexity=perp, random_state=42)
            X_tsne2 = tsne2.fit_transform(X_proj)
            plot_df['tSNE_2D_1'] = X_tsne2[:, 0]
            plot_df['tSNE_2D_2'] = X_tsne2[:, 1]

            fig_tsne2 = px.scatter(plot_df, x='tSNE_2D_1', y='tSNE_2D_2', color='Group', symbol='pred_class',
                                   hover_name='sub_id', title=f'Manifold t-SNE 2D ({model_name})')
            save_interactive_plot(fig_tsne2, "manifold_tsne2d")

            tsne3 = TSNE(n_components=3, perplexity=perp, random_state=42)
            X_tsne3 = tsne3.fit_transform(X_proj)
            plot_df['tSNE_3D_1'] = X_tsne3[:, 0]
            plot_df['tSNE_3D_2'] = X_tsne3[:, 1]
            plot_df['tSNE_3D_3'] = X_tsne3[:, 2]

            fig_tsne3 = px.scatter_3d(plot_df, x='tSNE_3D_1', y='tSNE_3D_2', z='tSNE_3D_3', color='Group',
                                      symbol='pred_class', hover_name='sub_id',
                                      title=f'Manifold t-SNE 3D ({model_name})')
            save_interactive_plot(fig_tsne3, "manifold_tsne3d")
        except Exception as e:
            logger.warning(f"t-SNE plotting failed: {e}")

        # PCA 2D/3D
        try:
            pca = PCA(n_components=3, random_state=42)
            X_pca = pca.fit_transform(X_proj)
            plot_df['PCA_1'] = X_pca[:, 0]
            plot_df['PCA_2'] = X_pca[:, 1]
            plot_df['PCA_3'] = X_pca[:, 2]

            fig_pca2 = px.scatter(plot_df, x='PCA_1', y='PCA_2', color='Group', symbol='pred_class',
                                  hover_name='sub_id', title=f'Manifold PCA 2D ({model_name})')
            save_interactive_plot(fig_pca2, "manifold_pca2d")

            fig_pca3 = px.scatter_3d(plot_df, x='PCA_1', y='PCA_2', z='PCA_3', color='Group', symbol='pred_class',
                                     hover_name='sub_id', title=f'Manifold PCA 3D ({model_name})')
            save_interactive_plot(fig_pca3, "manifold_pca3d")
        except Exception as e:
            logger.warning(f"PCA plotting failed: {e}")

    def build_pipeline(self, model):
        """Helper to build the scikit-learn pipeline."""
        steps = []
        if model == "LR":
            clf = make_pipeline(
                StandardScaler(),
                SelectKBest(score_func=f_classif, k=200),
                LogisticRegression(penalty="l2", solver="liblinear", max_iter=5000,
                                   class_weight="balanced")
            )
        elif model == "SVM":
            clf = make_pipeline(
                StandardScaler(),
                SelectKBest(score_func=f_classif, k=200),
                SVC(kernel="linear", probability=True)
            )
        elif model == "DTree":
            clf = make_pipeline(
                SelectFromModel(
                    RandomForestClassifier(n_estimators=100, random_state=42),
                    max_features=500
                ),
                DecisionTreeClassifier(max_depth=4, min_samples_leaf=5, random_state=42, class_weight="balanced")
            )
        elif model == "RandForest":
            clf = make_pipeline(
                SelectFromModel(
                    RandomForestClassifier(n_estimators=100, random_state=42),
                    max_features=10
                ),
                RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=2, random_state=42)
            )
        elif model == "NN":
            # Use the DeepSklearnWrapper (a minimal sklearn-like wrapper around a PyTorch model)
            # The wrapper supports fit/predict/predict_proba and exposes feature_importances_
            clf = make_pipeline(
                StandardScaler(),
                SelectKBest(score_func=f_classif, k=300),
                DeepSklearnWrapper(
                    epochs=120,
                    lr=0.001,
                    weight_decay=1e-3,
                    latent_dim=16,
                    n_ensembles=1,
                    seed=42,
                    patience=15,
                )
            )
        else:
            raise ValueError(f"Unsupported ML model: {model}")

        steps.append(("clf", clf))
        return Pipeline(steps)

    def get_params_from_json(self, model, prefix="clf__"):
        # Load hyperparameters from the JSON configuration file
        json_path = self.args.ml_hyperparameters_file
        with open(json_path, 'r') as file:
            all_hyperparameters = json.load(file)

        if model in all_hyperparameters:
            model_specific_params = all_hyperparameters[model]

            # Update the args namespace with the specific model's parameters
            for param_name, param_values in model_specific_params.items():
                # Replace python's null with None just in case, though sklearn handles None well
                clean_values = [None if v is None else v for v in param_values]
                setattr(self.args, f"ml_param_{param_name}", clean_values)

            # 3. Define the hyperparameter grid dynamically
            # Prepend the pipeline prefix to each parameter key automatically
            param_grid = {f"{prefix}{key}": getattr(self.args, f"ml_param_{key}")
                          for key in model_specific_params.keys()}
            return param_grid
        else:
            # Handle case where the model is not found in the JSON
            raise ValueError(f"Model '{model}' not found in {json_path}")

    def analyze(self, X, ids, atlas_labels, model, zero_class, one_class):
        self.logger.info(f"Analyzing model {model} pair: {zero_class} vs {one_class}")

        self.step_folder_groups = os.path.join(self.step_folder, f"ml_{zero_class}_vs_{one_class}")
        os.makedirs(self.step_folder_groups, exist_ok=True)
        data = self.prepare_data(ids, zero_class, one_class)

        X = X[data["x_row_idx"].values]
        y = data["y_classical_tendency"].astype(int).values

        if len(np.unique(y)) < 2:
            raise RuntimeError("Training labels for musicians contain only one class")

        # 1. Fetch the base pipeline
        pipe = self.build_pipeline(model)

        # ---------------------------------------------------------
        # 2. Setup the Base Estimator for Search (Robust to extreme imbalance)
        # ---------------------------------------------------------
        minority_class_size = np.min(np.bincount(y))

        # Only use bagging if the minority class is large enough to survive random bootstrapping
        is_bagging = (len(data) >= 20) and (minority_class_size >= 7)

        if is_bagging:
            # Wrap the pipeline in Bagging FIRST
            base_model_for_search = BalancedBaggingClassifier(
                estimator=pipe,
                n_estimators=30,
                sampling_strategy='majority',
                n_jobs=1,
                random_state=42
            )
            outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            # When wrapped, parameter keys must start with 'estimator__'
            prefix = "estimator__clf__"
        else:
            base_model_for_search = pipe
            outer_cv = LeaveOneOut()
            prefix = "clf__"

        param_grid = self.get_params_from_json(model, prefix)

        # ---------------------------------------------------------
        # 4. Setup the Inner CV (GridSearchCV) safely
        # ---------------------------------------------------------
        if minority_class_size >= 3:
            inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        else:
            inner_cv = LeaveOneOut()

        grid_search = GridSearchCV(
            estimator=base_model_for_search,
            param_grid=param_grid if param_grid else {},
            cv=inner_cv,
            scoring='balanced_accuracy',
            n_jobs=self.args.jobs,
            verbose=3  # Print real-time progress for each parameter combination
        )

        print("[ML] Starting Nested CV for performance evaluation...")
        outer_estimator = grid_search

        # Outer CV Evaluation
        y_pred = cross_val_predict(outer_estimator, X, y, cv=outer_cv, method="predict", verbose=10)
        try:
            y_prob = cross_val_predict(outer_estimator, X, y, cv=outer_cv, method="predict_proba", verbose=10)[:, 1]
        except Exception:
            y_prob = None

        acc = accuracy_score(y, y_pred)
        bacc = balanced_accuracy_score(y, y_pred)
        auc = roc_auc_score(y, y_prob) if y_prob is not None and len(np.unique(y)) == 2 else np.nan
        cm = confusion_matrix(y, y_pred)

        # 5. Final Fit on ALL data to extract the absolute best parameters
        print("[ML] Fitting final model on all data...")
        grid_search.fit(X, y)
        best_final_model = grid_search.best_estimator_

        if param_grid:
            print(f"[ML] Optimal parameters chosen: {grid_search.best_params_}")

        # -----------------------------
        # Permutation test (Evaluates the finalized best model only)
        # -----------------------------
        n_perm = getattr(self.args, "n_permutations", 5000)

        def run_single_permutation(perm_idx):
            """Helper function to run a single permutation."""
            start_time = time.time()
            print(f"[Perm-Worker] Starting permutation {perm_idx + 1}/{n_perm}")

            # Seed per iteration for reproducibility
            rng_local = np.random.default_rng(42 + perm_idx)
            y_perm = rng_local.permutation(y)

            try:
                # Run permutation strictly on the frozen best model (prevents re-searching)
                y_perm_pred = cross_val_predict(best_final_model, X, y_perm, cv=outer_cv, method="predict")
                score = balanced_accuracy_score(y_perm, y_perm_pred)
            except ValueError as e:
                print(f"[Perm-Worker] Skipping permutation {perm_idx + 1} due to bootstrap class imbalance.")
                return np.nan
            except Exception as e:
                print(f"[Perm-Worker] Skipping permutation {perm_idx + 1} due to error: {e}")
                return np.nan

            elapsed_time = time.time() - start_time
            print(f"[Perm-Worker] Completed permutation {perm_idx + 1}/{n_perm} | Duration: {elapsed_time:.2f}s")

            return score

        print(f"[ML] Running {n_perm} permutation tests in parallel ({getattr(self.args, 'jobs', 1)} jobs)...")
        perm_scores_raw = Parallel(n_jobs=getattr(self.args, 'jobs', 1), verbose=10)(
            delayed(run_single_permutation)(i) for i in range(n_perm)
        )

        perm_scores = np.array(perm_scores_raw, dtype=float)
        perm_scores = perm_scores[~np.isnan(perm_scores)]
        actual_n_perms = len(perm_scores)
        print(f"[ML] Successfully completed {actual_n_perms} out of {n_perm} permutations.")

        perm_p = (np.sum(perm_scores >= bacc) + 1) / (actual_n_perms + 1)
        print(f"[ML] Permutation test completed: p-value = {perm_p:.6f}")

        # 6. Predict Non-Musicians (NM)
        data_results = data.copy()
        data_results["cv_pred"] = y_pred
        if y_prob is not None:
            data_results["cv_prob"] = y_prob

        # ---------------------------------------------------------
        # Outputs, logging and robust inner-model extraction
        # ---------------------------------------------------------

        # Drill down safely through Bagging -> Imblearn Pipeline -> Our Pipeline
        extracted_model = best_final_model

        # 1. If it's a Bagging ensemble, take the first fitted estimator
        if hasattr(extracted_model, "estimators_") and len(extracted_model.estimators_) > 0:
            extracted_model = extracted_model.estimators_[0]

        # 2. Imblearn BalancedBagging wraps models in an internal pipeline with a 'classifier' step
        if hasattr(extracted_model, "named_steps") and "classifier" in extracted_model.named_steps:
            extracted_model = extracted_model.named_steps["classifier"]

        # 3. Finally, our own build_pipeline wraps the logic in a 'clf' step
        if hasattr(extracted_model, "named_steps") and "clf" in extracted_model.named_steps:
            final_clf_for_roi = extracted_model.named_steps["clf"]
        else:
            final_clf_for_roi = extracted_model

        # ==========================================
        # Extract Projection Space and Plot Manifolds
        # ==========================================
        oof_predictions = y_pred  # Already computed via CV above

        if model == "NN":
            try:
                # Use 'extracted_model' which is the unwrapped scikit-learn Pipeline
                # instead of 'best_final_model' which might be a Bagging wrapper
                X_preprocessed = extracted_model[:-1].transform(X)

                # Use the extracted final_clf_for_roi (the DeepSklearnWrapper)
                # to get the 16D latent space
                X_proj = final_clf_for_roi.transform_latent(X_preprocessed)
                self.logger.info(f"Successfully extracted latent space with shape: {X_proj.shape}")
            except Exception as e:
                self.logger.warning(f"Could not extract latent space, falling back to raw X: {e}")
                X_proj = X
        else:
            # For classic models, project the raw features
            X_proj = X

        try:
            # Call the clean plotting function
            self.plot_model_manifolds(X_proj, oof_predictions, data, model_name=model)
        except Exception as e:
            self.logger.warning(f"plot_model_manifolds failed: {e}")

        tag = now_str()
        cv_csv = os.path.join(self.step_folder_groups,
                              f"ml_cv_results_{model}_{self.args.pc_str}_{tag}.csv")
        report_txt = os.path.join(self.step_folder_groups, f"ml_report_{model}_{self.args.pc_str}_{tag}.txt")
        cm_png = os.path.join(self.step_folder_groups, f"ml_confusion_matrix_{model}_{self.args.pc_str}_{tag}.png")
        perm_png = os.path.join(self.step_folder_groups,
                                f"ml_permutation_test_{model}_{self.args.pc_str}_{tag}.png")

        data_results.to_csv(cv_csv, index=False)

        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("CV Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        savefig(cm_png)

        plt.figure(figsize=(6, 4))
        sns.histplot(perm_scores, bins=20, color="gray")
        plt.axvline(bacc, color="red", linestyle="--", label=f"Observed BAcc = {bacc:.3f}")
        plt.xlabel("Balanced Accuracy under permutation")
        plt.ylabel("Count")
        plt.title("Permutation Test")
        plt.legend()
        savefig(perm_png)

        with open(report_txt, "w", encoding="utf-8") as f:
            f.write("=" * 88 + "\n")
            f.write("ML REPORT\n")
            f.write("=" * 88 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"ML model: {model}\n")
            if param_grid:
                f.write(f"Best Tuned Parameters: {grid_search.best_params_}\n")
            f.write(f"Target PC index: {self.args.pc_str}\n")
            f.write(f"Row normalization: {self.args.row_normalize_ml}\n")
            f.write(f"Permutations: {n_perm}\n\n")

            f.write("--- TRAIN SET ---\n")
            f.write(f"Number of samples: {len(data)}\n")
            f.write(data["y_classical_tendency"].value_counts().to_string() + "\n\n")

            f.write("--- CV PERFORMANCE ---\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Balanced Accuracy: {bacc:.4f}\n")
            f.write(f"ROC AUC: {auc if not np.isnan(auc) else 'NA'}\n")
            f.write("Confusion matrix:\n")
            f.write(np.array2string(cm) + "\n\n")
            f.write(classification_report(y, y_pred) + "\n")

            f.write("--- PERMUTATION TEST ---\n")
            f.write(f"Observed balanced accuracy: {bacc:.4f}\n")
            if actual_n_perms > 0:
                f.write(f"Permutation mean: {np.mean(perm_scores):.4f}\n")
                f.write(f"Permutation std: {np.std(perm_scores, ddof=1):.4f}\n")
            f.write(f"Permutation p-value: {perm_p:.6f}\n\n")
        # ==========================================
        # Save the FULL trained pipeline for future test sets
        # ==========================================

        # Define the save path (.pkl extension is standard for pickled models)
        model_pkl_path = os.path.join(self.step_folder_groups, f"best_model_pipeline_{model}_{self.args.pc_str}_{tag}.pkl")

        try:
            joblib.dump(best_final_model, model_pkl_path)
            self.logger.info(f"Successfully saved full model pipeline to {model_pkl_path}")
        except Exception as e:
            self.logger.error(f"Failed to save model pipeline: {e}")
            model_pkl_path = None

        return {
            "fitted_clf": final_clf_for_roi,
            "atlas_labels": atlas_labels,
            "cv_csv": cv_csv,
            "report_txt": report_txt,
            "perm_p": perm_p,
            "model_path": model_pkl_path,  # Added the saved model path to the output
            "step_folder_groups": self.step_folder_groups
        }

