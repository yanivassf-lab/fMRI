import os
import sys
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class Statistics():
    def __init__(self, base_directory):
        self.base_directory = base_directory
        self.output_directory = os.path.join(self.base_directory, "global_summary")


    def parse_report_file(self, filepath):
        """
        Reads a single ML report text file and extracts key metrics using regex.
        Returns a dictionary with the extracted data.
        """
        data = {
            "ml_model": None,
            "balanced_accuracy": None,
            "roc_auc": None,
            "p_value": None,
            "nm_prop_1": None
        }

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

            # Extract ML Model
            model_match = re.search(r"ML model:\s+(\w+)", content)
            if model_match:
                model = model_match.group(1)
                if model in ["LR", "SVM", "DTree", "RandForest", "NN"]:
                    data["ml_model"] = model

            # Extract Balanced Accuracy
            bacc_match = re.search(r"Balanced Accuracy:\s+([0-9\.]+)", content)
            if bacc_match:
                data["balanced_accuracy"] = float(bacc_match.group(1))

            # Extract ROC AUC
            auc_match = re.search(r"ROC AUC:\s+([0-9\.]+)", content)
            if auc_match:
                data["roc_auc"] = float(auc_match.group(1))

            # Extract Permutation p-value
            pval_match = re.search(r"Permutation p-value:\s+([0-9\.]+)", content)
            if pval_match:
                data["p_value"] = float(pval_match.group(1))

            # Extract NM prediction proportion for class 1
            nm_match = re.search(r"NM predicted 1 proportion:\s+([0-9\.]+)", content)
            if nm_match:
                data["nm_prop_1"] = float(nm_match.group(1))

        return data


    def compile_results(self):
        """
        Walks through the directory structure, finds all relevant report files,
        and compiles them into a single pandas DataFrame.
        """
        results = []

        # Find all directories matching the pair pattern
        pair_dirs = glob.glob(os.path.join(self.base_directory, "ml_*_vs_*"))

        for pair_dir in pair_dirs:
            folder_name = os.path.basename(pair_dir)
            # Extract the groups from the folder name
            match = re.match(r"ml_(.+?)_vs_(.+)", folder_name)
            if not match:
                continue

            group_0, group_1 = match.groups()
            pair_name = f"{group_0} vs {group_1}"

            # Find all report txt files in this directory
            report_files = glob.glob(os.path.join(pair_dir, "ml_report_*.txt"))

            for report_file in report_files:
                file_data = self.parse_report_file(report_file)

                # Only add if we successfully extracted the model name
                if file_data["ml_model"]:
                    file_data["pair"] = pair_name
                    file_data["group_0"] = group_0
                    file_data["group_1"] = group_1
                    results.append(file_data)

        return pd.DataFrame(results)


    def generate_visualizations(self, df):
        """
        Generates and saves heatmap visualizations for Balanced Accuracy and p-values.
        """
        if df.empty:
            print("No data found to visualize.")
            return

        # Ensure output directory exists
        os.makedirs(self.output_directory, exist_ok=True)

        # Pivot table for Balanced Accuracy
        bacc_pivot = df.pivot_table(
            index="pair",
            columns="ml_model",
            values="balanced_accuracy",
            aggfunc="max"
        )

        # Plot 1: Balanced Accuracy Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(bacc_pivot, annot=True, cmap="YlGnBu", fmt=".3f", vmin=0.4, vmax=1.0)
        plt.title("Balanced Accuracy by Pair and Model")
        plt.ylabel("Instrument Pair")
        plt.xlabel("ML Model")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_directory, "summary_balanced_accuracy.png"), dpi=300)
        plt.close()

        # Pivot table for p-values (if permutation tests were run)
        if not df["p_value"].isna().all():
            pval_pivot = df.pivot_table(
                index="pair",
                columns="ml_model",
                values="p_value",
                aggfunc="min"
            )

            # Plot 2: P-Value Heatmap
            plt.figure(figsize=(10, 8))
            # Using a custom colormap where lower values (good) are darker
            sns.heatmap(pval_pivot, annot=True, cmap="Reds_r", fmt=".3f", vmax=0.1)
            plt.title("Permutation P-Value by Pair and Model")
            plt.ylabel("Instrument Pair")
            plt.xlabel("ML Model")
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_directory, "summary_p_values.png"), dpi=300)
            plt.close()


    def create_summary_files(self):
        # Set this to the directory containing all the ml_X_vs_Y folders
        # Assuming the script is run from the parent directory
        # base_directory = "/Users/user/Documents/pythonProject/fMRI-runs/full_data/analyses/group_run/full_pipeline_set2/2_ml_pc-1"

        print("Scanning directories for ML reports...")
        df_results = self.compile_results()

        if df_results.empty:
            print("No results found. Please check the directory path and folder structures.")
            return

        # Reorder columns for readability
        cols = ["pair", "ml_model", "balanced_accuracy", "roc_auc", "p_value", "nm_prop_1", "group_0", "group_1"]
        df_results = df_results[cols]

        # Sort by Balanced Accuracy (highest first)
        df_results = df_results.sort_values(by="balanced_accuracy", ascending=False)

        # Save to CSV
        csv_path = os.path.join(self.output_directory, "all_pairs_summary.csv")
        os.makedirs(self.output_directory, exist_ok=True)
        df_results.to_csv(csv_path, index=False)
        print(f"Summary CSV saved to: {csv_path}")

        # Generate Plots
        print("Generating visualizations...")
        self.generate_visualizations(df_results)
        print(f"Visualizations saved in the '{self.output_directory}' folder.")

def main(base_dir):
    stat = Statistics(base_dir)
    stat.create_summary_files()

if __name__ == '__main__':
    base_dir = sys.argv[1]
    main(base_dir)
