import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib

# This code fix old pkl files that were saved with the old package structure. It ensures that when loading these models, the necessary classes are available in the expected module paths.
# import sys
# import types
#
# import PCAGroupDiscrimination as pkg
#
# from PCAGroupDiscrimination.cli.nn_wrapper import (
#     DeepSklearnWrapper,
#     LatentSpaceNN
# )
#
# old_root = types.ModuleType("PCACorrelator")
# old_root.__path__ = []
#
# # cli package
# old_cli = types.ModuleType("PCACorrelator.cli")
# old_cli.__path__ = []
#
# # unified_pipeline module
# old_up = types.ModuleType("PCACorrelator.cli.unified_pipeline")
#
# old_up.DeepSklearnWrapper = DeepSklearnWrapper
# old_up.LatentSpaceNN = LatentSpaceNN
#
# old_cli.unified_pipeline = old_up
# old_root.cli = old_cli
#
# sys.modules["PCACorrelator"] = old_root
# sys.modules["PCACorrelator.cli"] = old_cli
# sys.modules["PCACorrelator.cli.unified_pipeline"] = old_up




matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix, \
    classification_report

from .fmri_fpca_pipeline import setup_logger
from .utils import now_str, savefig

# Crucial import: Required for joblib to successfully unpickle the custom NN model.
# Using noqa to prevent IDE warnings about unused imports.
from .nn_wrapper import DeepSklearnWrapper  # noqa: F401


class MLTester:
    def __init__(self, args):
        self.args = args
        # The directory containing the trained models
        self.train_folder = os.path.join(self.args.output_dir, f"2_ml_{args.pc_str}")

        # New dedicated directory for test results
        self.step_folder = os.path.join(self.args.output_dir, f"3_ml_test_{args.pc_str}")
        os.makedirs(self.step_folder, exist_ok=True)
        self.logger = setup_logger(self.step_folder)

    def calculate_binary_class_test(self, row, zero_class=None, one_class=None):
        ins_group = row["instrument_group"]
        is_train_set = row["is_train_set"]

        # Core requirement: Only subjects with is_train_set == 0 belong to the test set
        if is_train_set != 0:
            return np.nan

        if (zero_class == "NM" and one_class == "mus") or (zero_class == "mus" and one_class == "NM"):
            classification = 0 if ins_group == "NM" else 1
        else:
            classification = 0 if ins_group == zero_class else 1 if ins_group == one_class else np.nan

        return classification

    def prepare_data_test(self, ids, zero_class, one_class):
        feat = pd.read_csv(self.args.metadata_csv)

        # Clean up unnamed columns
        unnamed_cols = [c for c in feat.columns if str(c).startswith("Unnamed:")]
        if unnamed_cols:
            feat = feat.drop(columns=unnamed_cols)

        feat["instrument_group"] = feat["instrument_group"].fillna("NM")
        feat["y_classical_tendency"] = feat.apply(
            self.calculate_binary_class_test,
            axis=1,
            args=(zero_class, one_class)
        )

        # Merge extracted coordinates with the metadata labels
        Xmeta = pd.DataFrame({"sub_id": ids, "x_row_idx": np.arange(len(ids))})
        merged = Xmeta.merge(
            feat[["sub_id", "group", "instrument_group", "y_classical_tendency"]],
            on="sub_id", how="inner"
        )
        return merged[merged["y_classical_tendency"].notnull()]

    def test(self, X, ids, model_name, zero_class, one_class):
        self.logger.info(f"Testing model {model_name} on TEST SET for pair: {zero_class} vs {one_class}")

        self.step_folder_groups = os.path.join(self.step_folder, f"ml_test_{zero_class}_vs_{one_class}")
        os.makedirs(self.step_folder_groups, exist_ok=True)

        # Fetch test data
        data = self.prepare_data_test(ids, zero_class, one_class)
        if data.empty:
            self.logger.warning(f"No test data available for pair {zero_class} vs {one_class}. Skipping.")
            return

        # Extract strictly the test samples
        X_test = X[data["x_row_idx"].values]
        y_test = data["y_classical_tendency"].astype(int).values

        # Ensure both classes are present in the test set to calculate meaningful metrics
        if len(np.unique(y_test)) < 2:
            self.logger.warning(f"Test set for {zero_class} vs {one_class} contains only one class. Skipping metrics.")
            return

        # ---------------------------------------------------------
        # Locate and load the trained model from the training directory
        # ---------------------------------------------------------
        train_subfolder = os.path.join(self.train_folder, f"ml_{zero_class}_vs_{one_class}")
        search_pattern = os.path.join(train_subfolder, f"best_model_pipeline_{model_name}_{self.args.pc_str}_*.pkl")
        model_files = glob.glob(search_pattern)

        if not model_files:
            self.logger.error(f"Could not find trained .pkl model for {model_name}. Did the training phase complete?")
            return

        # Select the most recently created model file in case multiple exist
        latest_model_path = max(model_files, key=os.path.getmtime)
        self.logger.info(f"Loaded trained model from: {latest_model_path}")

        loaded_pipeline = joblib.load(latest_model_path)

        # ---------------------------------------------------------
        # Execute predictions on the unseen test set
        # The pipeline automatically applies scaling and SelectKBest
        # ---------------------------------------------------------
        y_pred = loaded_pipeline.predict(X_test)
        try:
            y_prob = loaded_pipeline.predict_proba(X_test)[:, 1]
        except Exception:
            y_prob = None

        acc = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else np.nan
        cm = confusion_matrix(y_test, y_pred)

        # ---------------------------------------------------------
        # Save results, metrics, and figures
        # ---------------------------------------------------------
        tag = now_str()
        data_results = data.copy()
        data_results["test_pred"] = y_pred
        if y_prob is not None:
            data_results["test_prob"] = y_prob

        csv_out = os.path.join(self.step_folder_groups, f"test_results_{model_name}_{self.args.pc_str}_{tag}.csv")
        cm_png = os.path.join(self.step_folder_groups,
                              f"test_confusion_matrix_{model_name}_{self.args.pc_str}_{tag}.png")
        report_txt = os.path.join(self.step_folder_groups, f"test_report_{model_name}_{self.args.pc_str}_{tag}.txt")

        data_results.to_csv(csv_out, index=False)

        # Generate confusion matrix visualization
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
        plt.title("Test Set Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        savefig(cm_png)

        # Generate textual performance report
        with open(report_txt, "w", encoding="utf-8") as f:
            f.write("=" * 88 + "\n")
            f.write("TEST SET REPORT\n")
            f.write("=" * 88 + "\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"ML model: {model_name}\n")
            f.write(f"Target PC index: {self.args.pc_str}\n\n")

            f.write("--- TEST SET INFO ---\n")
            f.write(f"Number of test samples: {len(data)}\n")
            f.write(data["y_classical_tendency"].value_counts().to_string() + "\n\n")

            f.write("--- TEST PERFORMANCE ---\n")
            f.write(f"Accuracy: {acc:.4f}\n")
            f.write(f"Balanced Accuracy: {bacc:.4f}\n")
            f.write(f"ROC AUC: {auc if not np.isnan(auc) else 'NA'}\n")
            f.write("Confusion matrix:\n")
            f.write(np.array2string(cm) + "\n\n")
            f.write(classification_report(y_test, y_pred) + "\n")

        self.logger.info(f"Test complete. BAcc: {bacc:.3f}")
        return
