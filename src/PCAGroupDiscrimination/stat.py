import os
import sys
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr


class Statistics():
    def __init__(self, base_directory, align_dir, pc_num=None):
        self.base_directory = base_directory
        self.align_dir = align_dir
        self.output_directory = os.path.join(self.base_directory, "global_summary")
        self.pc_num = pc_num

    def parse_report_file(self, filepath):
        """
        Reads a single ML report text file and extracts key metrics using regex.
        Returns a dictionary with the extracted data.
        Note: ML p-value is parsed but no longer visualized as per request.
        """
        data = {
            "ml_model": None,
            "balanced_accuracy": None,
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

    def process_rsa_for_pairs(self, df_ml, df_rsa):
        """
        Matches RSA metrics (inter-distance, p/q values, intra-distances) to ML pairs.
        Calculates averages for 'nm vs mus'.
        """
        rsa_dict = {}
        if df_rsa is not None and not df_rsa.empty:
            for _, row in df_rsa.iterrows():
                pair = str(row['pair'])
                parts = [p.strip() for p in pair.split('vs')]

                if len(parts) == 2:
                    # Map exactly as it appears in RSA
                    rsa_dict[pair] = {
                        'distance': row['distance'],
                        'rsa_p': row['rsa_p'],
                        'rsa_q': row['rsa_q'],
                        'group_0_name': row['group_a_name'],
                        'group_0_intra': row['group_a_intra'],
                        'group_1_name': row['group_b_name'],
                        'group_1_intra': row['group_b_intra']
                    }

                    # Map reverse direction manually handling intra mapping
                    reversed_pair = f"{parts[1]} vs {parts[0]}"
                    rsa_dict[reversed_pair] = {
                        'distance': row['distance'],
                        'rsa_p': row['rsa_p'],
                        'rsa_q': row['rsa_q'],
                        'group_0_name': row['group_b_name'],
                        'group_0_intra': row['group_b_intra'],
                        'group_1_name': row['group_a_name'],
                        'group_1_intra': row['group_a_intra']
                    }

        # Initialize lists for DataFrame columns
        metrics = {'distance': [], 'rsa_p': [], 'rsa_q': [],
                   'group_0_name': [], 'group_0_intra': [],
                   'group_1_name': [], 'group_1_intra': []}

        for _, row in df_ml.iterrows():
            pair = row['pair']
            if pair in rsa_dict:
                for k in metrics.keys():
                    metrics[k].append(rsa_dict[pair][k])
            else:
                for k in metrics.keys():
                    metrics[k].append(np.nan if 'name' not in k else "N/A")

        # Assign back to DataFrame
        for k, v in metrics.items():
            df_ml[k] = v

        # Average handling for NM vs MUS
        nm_mus_mask = df_ml['pair'].str.lower().isin(["nm vs mus", "mus vs nm"])
        if nm_mus_mask.any() and rsa_dict:
            valid_dists, valid_ps, valid_qs = [], [], []
            valid_nm_intra, valid_mus_intra = [], []

            for k, v in rsa_dict.items():
                words = k.lower().split()
                # Find pairs where nm is compared to an instrument (not mus itself)
                if 'nm' in words and 'mus' not in words:
                    valid_dists.append(v['distance'])
                    valid_ps.append(v['rsa_p'])
                    valid_qs.append(v['rsa_q'])

                    # Identify which group was NM and get intra distances
                    if v['group_0_name'].lower() == 'nm':
                        valid_nm_intra.append(v['group_0_intra'])
                        valid_mus_intra.append(v['group_1_intra'])
                    elif v['group_1_name'].lower() == 'nm':
                        valid_nm_intra.append(v['group_1_intra'])
                        valid_mus_intra.append(v['group_0_intra'])

            if valid_dists:
                df_ml.loc[nm_mus_mask, 'distance'] = np.nanmean(valid_dists)
                df_ml.loc[nm_mus_mask, 'rsa_p'] = np.nanmean(valid_ps)
                df_ml.loc[nm_mus_mask, 'rsa_q'] = np.nanmean(valid_qs)
                df_ml.loc[nm_mus_mask, 'group_0_intra'] = np.nanmean(valid_nm_intra)
                df_ml.loc[nm_mus_mask, 'group_1_intra'] = np.nanmean(valid_mus_intra)
                df_ml.loc[nm_mus_mask, 'group_0_name'] = "NM"
                df_ml.loc[nm_mus_mask, 'group_1_name'] = "MUS_avg"

        return df_ml

    def generate_visualizations(self, df):
        if df.empty:
            print("No data found to visualize.")
            return

        os.makedirs(self.output_directory, exist_ok=True)
        bacc_corrs = {}
        top_3_scores = {}

        # Pivot table for Balanced Accuracy
        bacc_pivot = df.pivot_table(
            index="pair",
            columns="ml_model",
            values="balanced_accuracy",
            aggfunc="max"
        )

        top_3_dist_pairs = set(df.drop_duplicates('pair').nlargest(3, 'distance')['pair'])

        # Build informative Y labels containing Inter and Intra metrics
        y_labels_bacc = []
        for pair in bacc_pivot.index:
            row = df[df["pair"] == pair].iloc[0]
            dist = row['distance']
            p_val = row['rsa_p']
            q_val = row['rsa_q']
            g0_name = str(row['group_0_name'])[:6]
            g0_intra = row['group_0_intra']
            g1_name = str(row['group_1_name'])[:6]
            g1_intra = row['group_1_intra']

            if pd.notna(dist):
                label = (f"{pair}\n"
                         f"Dist: {dist:.3f} (p: {p_val:.3f}, q: {q_val:.3f})\n"
                         f"Intra: {g0_name}({g0_intra:.3f}), {g1_name}({g1_intra:.3f})")
            else:
                label = f"{pair}\n(RSA metrics missing)"
            y_labels_bacc.append(label)

        x_labels_bacc = []
        for col in bacc_pivot.columns:
            model_df = df[df['ml_model'] == col]

            top_3_bacc_pairs = set(model_df.nlargest(3, 'balanced_accuracy')['pair'])

            overlap_score = len(top_3_dist_pairs.intersection(top_3_bacc_pairs))
            top_3_scores[col] = overlap_score

            valid_data = model_df[['balanced_accuracy', 'distance']].dropna()
            if len(valid_data) > 1 and valid_data['distance'].nunique() > 1:
                corr, _ = pearsonr(valid_data['distance'], valid_data['balanced_accuracy'])
                x_labels_bacc.append(f"{col}\nCorr: {corr:.3f}\nTop3 Match: {overlap_score}/3")
                bacc_corrs[col] = corr
            else:
                x_labels_bacc.append(f"{col}\nCorr: N/A\nTop3 Match: {overlap_score}/3")
                bacc_corrs[col] = np.nan

        # Adjust figure size dynamically based on number of pairs
        fig_height = max(10, len(bacc_pivot) * 1.5)
        plt.figure(figsize=(16, fig_height))

        ax1 = sns.heatmap(bacc_pivot, annot=True, cmap="YlGnBu", fmt=".3f", vmin=0.4, vmax=1.0)
        ax1.set_yticklabels(y_labels_bacc, rotation=0, fontsize=10)
        ax1.set_xticklabels(x_labels_bacc, rotation=0, fontsize=11)  # הקטנת פונט מעט כדי להכיל 3 שורות

        plt.title("Balanced Accuracy by Pair and Model", fontsize=18, pad=20)
        plt.ylabel("Instrument Pair (with RSA Metrics)", fontsize=14, labelpad=15)
        plt.xlabel("ML Model", fontsize=14, labelpad=15)

        plt.subplots_adjust(left=0.35, bottom=0.18, top=0.95)  # הגדלת המרווח התחתון עבור הציון החדש
        plt.savefig(os.path.join(self.output_directory, "summary_balanced_accuracy.png"), dpi=300, bbox_inches='tight')
        plt.close()

        txt_path = os.path.join(self.output_directory, "correlations_summary.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("=== ML Models Alignment with Furthest Pairs ===\n")
            for model in bacc_pivot.columns:
                corr_str = f"{bacc_corrs[model]:.4f}" if pd.notna(bacc_corrs[model]) else "N/A"
                f.write(
                    f"Model: {model: <12} | Top-3 Overlap Score: {top_3_scores[model]}/3 | Global Corr: {corr_str}\n")

        print(f"Correlation and alignment summary saved to: {txt_path}")

    def create_summary_files(self):
        print(f"Scanning directories {self.align_dir} for ML reports...")
        df_results = self.compile_results()

        if df_results.empty:
            print("No results found. Please check the directory path and folder structures.")
            return

        # Load RSA metrics using PC number
        df_rsa = None
        if self.pc_num:
            search_pattern = os.path.join(self.align_dir, f"rsa_metrics_summary_{self.pc_num}_*.csv")
            matching_files = glob.glob(search_pattern, recursive=True)
            if matching_files:
                latest_file = max(matching_files, key=os.path.getmtime)
                df_rsa = pd.read_csv(latest_file)
                print(f"Loaded latest RSA metrics for {self.pc_num} from: {latest_file}")
            else:
                print(f"Notice: RSA metrics CSV for {self.pc_num} not found. Distances will be missing.")

        df_results = self.process_rsa_for_pairs(df_results, df_rsa)

        # Reorder columns for readability
        cols = ["pair", "ml_model", "balanced_accuracy", "distance", "rsa_p", "rsa_q",
                "group_0_name", "group_1_name", "group_0_intra", "group_1_intra",
                "nm_prop_1", "group_0", "group_1"]
        df_results = df_results[[c for c in cols if c in df_results.columns]]

        df_results = df_results.sort_values(by="balanced_accuracy", ascending=False)

        csv_path = os.path.join(self.output_directory, "all_pairs_summary.csv")
        os.makedirs(self.output_directory, exist_ok=True)
        df_results.to_csv(csv_path, index=False)
        print(f"Summary CSV saved to: {csv_path}")

        print("Generating visualization...")
        self.generate_visualizations(df_results)
        print(f"Visualization saved in the '{self.output_directory}' folder.")


def main(base_dir, align_dir, pc_str):
    stat = Statistics(base_dir, align_dir, pc_str)
    stat.create_summary_files()

if __name__ == '__main__':
    base_dir = sys.argv[1]
    align_dir = sys.argv[2]
    pc_str = sys.argv[3]
    main(base_dir, align_dir, pc_str)
