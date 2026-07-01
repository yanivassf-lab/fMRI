import os
import pandas as pd
import numpy as np
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import spearmanr
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

from .build_signals_from_files import setup_logger

class SignalAligner:
    def __init__(self, args):
        self.args = args
        self.step_prev_folder = os.path.join(self.args.output_dir, '1_align_signals')
        self.step_folder = os.path.join(self.args.output_dir, '1_align_signals')
        os.makedirs(self.step_folder, exist_ok=True)
        self.logger = setup_logger(self.step_folder)

    def align(self, X, ids, atlas_labels):
        self.plot_flattened_alignment(X, ids, pd.read_csv(self.args.metadata_csv),
                                      atlas_labels=atlas_labels)
        self.run_rsa_analysis(X, ids, pd.read_csv(self.args.metadata_csv))

    def plot_flattened_alignment(self, X, ids, metadata_df, atlas_labels):
        """
        Calculates signal variance once per PC to save a stats file,
        then plots the flattened feature alignment dynamically sorted by multiple columns.
        """
        # 1. Clean the sub_id column in CSV globally
        metadata_df['sub_id'] = metadata_df['sub_id'].astype(str).str.replace("sub-", "").str.strip()

        # Clean the input ids globally so merging and matching works perfectly
        clean_ids = [str(val).replace("sub-", "").strip() for val in ids]

        # =================================================================
        # NEW: Calculate and save Stats ONLY ONCE (Outside the loop)
        # =================================================================
        self.logger.info(f"Calculating global signal variances for {self.args.pc_str}...")

        # Calculate intensity on the raw X matrix
        row_intensities = np.mean(np.abs(X), axis=1)

        # Create a temporary dataframe for stats
        stats_df = pd.DataFrame({
            'sub_id': clean_ids,
            'row_intensity': row_intensities
        })

        cols_to_check = ['has_classical', 'has_jazz', 'has_rock', 'has_pop', 'is_wind', 'is_rhythm']
        available_cols = [c for c in cols_to_check if c in metadata_df.columns]

        # Merge will now work because both DataFrames use clean, prefix-free IDs
        merged_stats = stats_df.merge(metadata_df[['sub_id', 'group'] + available_cols], on='sub_id', how='left')
        top_dark_rows = merged_stats.sort_values(by='row_intensity', ascending=False).head(20)

        def is_positive_trait(v):
            if pd.isna(v): return False
            return str(v).strip().lower() in ['1', '1.0', 'true', 'yes', 't', 'y']

        for _, row in top_dark_rows.iterrows():
            # Get group if it exists, else Unknown
            traits = []
            for c in available_cols:
                if is_positive_trait(row[c]):
                    traits.append(c.replace('has_', '').replace('is_', '').upper())

        # =================================================================
        # Plotting Loop
        # =================================================================
        for target_col in [['is_rhythm', 'has_classical'], 'group', 'main_instrument', 'instrument_group', 'genres',
                           'has_classical', 'has_jazz',
                           'has_rock', 'has_pop', 'is_wind', 'is_rhythm', 'pca_tSNE']:

            # Generate a string-safe version of the column name for logging and file paths
            if isinstance(target_col, list):
                col_name_str = "_and_".join(target_col)
            else:
                col_name_str = str(target_col)

            self.logger.info(f"Generating flattened alignment visualization sorted by: {col_name_str}")

            records = []
            unknown_count = 0

            # Iterate over clean_ids to match the cleaned metadata
            for i, clean_sub in enumerate(clean_ids):
                matched_rows = metadata_df[metadata_df['sub_id'] == clean_sub]

                if not matched_rows.empty:
                    if isinstance(target_col, list):
                        # Extract values for multiple columns
                        raw_vals = matched_rows[target_col].iloc[0]

                        # Check if any of the requested columns have NaN/Missing values
                        if raw_vals.isna().any():
                            category = 'Unknown/None'
                        else:
                            # Combine the values into a single readable string (e.g., "1 | 0" or "True | False")
                            category = " | ".join([f"{col}: {val}" for col, val in zip(target_col, raw_vals)])
                    else:
                        # Original logic for a single column
                        raw_val = matched_rows[target_col].values[0]
                        category = str(raw_val).strip()
                        if pd.isna(raw_val) or category.lower() in ['nan', 'none', '']:
                            category = 'Unknown/None'
                else:
                    category = 'Missing Data'
                    unknown_count += 1

                records.append({
                    'sub_id': clean_sub,
                    'category': category,
                    'signal': X[i]
                })

            df = pd.DataFrame(records)
            df = df.sort_values(by=['category', 'sub_id']).reset_index(drop=True)
            X_sorted = np.vstack(df['signal'].values)

            X_contrast = X_sorted
            vmax = np.percentile(np.abs(X_contrast), 95)
            vmin = -vmax

            unique_categories = df['category'].unique()
            if len(unique_categories) <= 10:
                base_colors = plt.get_cmap('tab10').colors
            else:
                base_colors = plt.get_cmap('tab20').colors

            category_color_map = {cat: base_colors[i % len(base_colors)] for i, cat in enumerate(unique_categories)}

            # ---------------- Plotting ----------------
            fig, ax = plt.subplots(figsize=(20, 35))

            sns.heatmap(X_contrast,
                        cmap='RdBu_r',
                        center=0,
                        vmin=vmin,
                        vmax=vmax,
                        xticklabels=False,
                        yticklabels=df['sub_id'],
                        ax=ax,
                        cbar_kws={'label': 'Signal Amplitude', 'shrink': 0.5})

            n_regions = len(atlas_labels)
            n_timepoints = X_contrast.shape[1] // n_regions
            time_step = 10

            x_ticks = []
            x_labels = []

            for t in range(0, n_timepoints, time_step):
                start_idx = t * n_regions
                ax.axvline(start_idx, color='white', linewidth=0.8, alpha=0.6, linestyle='--')
                x_ticks.append(start_idx + (n_regions / 2))
                x_labels.append(f"TR {t}")

            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=12)

            current_category = df['category'].iloc[0]
            y_ticks_positions = []
            y_ticks_labels = []
            category_start_idx = 0

            for i, category in enumerate(df['category']):
                if category != current_category:
                    ax.axhline(i, color='white', linewidth=3)
                    y_ticks_positions.append(category_start_idx + (i - category_start_idx) / 2)
                    y_ticks_labels.append(current_category)
                    current_category = category
                    category_start_idx = i

            y_ticks_positions.append(category_start_idx + (len(df) - category_start_idx) / 2)
            y_ticks_labels.append(current_category)

            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim())
            ax2.set_yticks(y_ticks_positions)

            short_labels = [str(lbl)[:25] + '...' if len(str(lbl)) > 25 else str(lbl) for lbl in y_ticks_labels]
            ax2.set_yticklabels(short_labels, fontsize=18, fontweight='bold')

            for tick_label, original_label in zip(ax2.get_yticklabels(), y_ticks_labels):
                correct_color = category_color_map.get(original_label, 'black')
                tick_label.set_color(correct_color)
                tick_label.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.8, pad=3))

            ax.set_xlabel("Concatenated Spatiotemporal Features (Time x Regions)", fontsize=16, fontweight='bold')
            ax.set_ylabel("Subjects", fontsize=16, fontweight='bold')

            # Use col_name_str for the title
            ax.set_title(f"Feature Alignment - Grouped by '{col_name_str}'\n({self.args.pc_str})",
                         fontsize=22, pad=20)
            ax.tick_params(axis='y', labelsize=8)

            plt.tight_layout()

            # Use col_name_str for the output path to avoid filesystem errors
            out_image_path = os.path.join(self.step_folder,
                                          f"flattened_alignment_{col_name_str}_{self.args.pc_str}.png")
            plt.savefig(out_image_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved plot to {out_image_path}")
            plt.close()

    def run_rsa_analysis(self, X, ids, metadata_df):
        """
        Runs Representational Similarity Analysis (RSA) on the concatenated features.
        Uses data cleaning, matches IDs, displays category labels on the heatmaps,
        computes group-to-group proximity, clusters groups into 2 main clusters.
        Applies False Discovery Rate (FDR) correction globally and locally.

        STATISTICAL LOGIC:
        - Intra-group (Diagonal): Tests for COHESION (true distance < chance).
        - Inter-group (Off-diagonal): Tests for SEPARATION (true distance > chance),
          which is ideal for downstream Machine Learning separation.
        """

        # Helper function for False Discovery Rate (Benjamini-Hochberg)
        def fdr_bh(pvals):
            """ Computes FDR q-values from a list of p-values. """
            pvals = np.asarray(pvals)
            n = len(pvals)
            if n == 0:
                return np.array([])
            order = np.argsort(pvals)
            ranks = np.arange(1, n + 1)
            qvals = np.zeros(n)
            qvals[order] = pvals[order] * n / ranks
            for i in range(n - 2, -1, -1):
                qvals[order[i]] = min(qvals[order[i]], qvals[order[i + 1]])
            return np.minimum(qvals, 1.0)

        # 1. Clean the sub_id column in CSV globally
        metadata_df['sub_id'] = metadata_df['sub_id'].astype(str).str.replace("sub-", "").str.strip()

        # Clean the input ids globally so merging and matching works perfectly
        clean_ids = [str(val).replace("sub-", "").strip() for val in ids]

        # Define output text file path once per run
        tm = datetime.now().strftime("%Y%m%d_%H%M%S")
        stats_out_path = os.path.join(self.step_folder, f"rsa_summary_{self.args.pc_str}_{tm}.txt")

        # Helper function for global permutation test
        def permutation_test_rsa(brain_dist_vec, labels, n_permutations=1000):
            """
            Performs global RSA permutation test by shuffling category labels.
            """
            n_subj = len(labels)
            true_model_rdm = np.zeros((n_subj, n_subj))
            for i in range(n_subj):
                for j in range(i + 1, n_subj):
                    if labels[i] != labels[j]:
                        true_model_rdm[i, j] = 1
                        true_model_rdm[j, i] = 1
            true_model_dist_vec = squareform(true_model_rdm)
            original_rho, _ = spearmanr(brain_dist_vec, true_model_dist_vec)

            permuted_rhos = []
            for _ in range(n_permutations):
                shuffled_labels = np.random.permutation(labels)
                perm_model_rdm = np.zeros((n_subj, n_subj))
                for i in range(n_subj):
                    for j in range(i + 1, n_subj):
                        if shuffled_labels[i] != shuffled_labels[j]:
                            perm_model_rdm[i, j] = 1
                            perm_model_rdm[j, i] = 1
                perm_model_dist_vec = squareform(perm_model_rdm)
                perm_rho, _ = spearmanr(brain_dist_vec, perm_model_dist_vec)
                permuted_rhos.append(perm_rho)

            permuted_rhos = np.array(permuted_rhos)
            p_value = (np.sum(permuted_rhos >= original_rho) + 1) / (n_permutations + 1)
            return original_rho, p_value, permuted_rhos, true_model_rdm

        # =================================================================
        # Phase 1: Collect Data for all Targets
        # =================================================================
        target_columns = [['is_rhythm', 'has_classical'], 'group', 'main_instrument', 'instrument_group', 'genres',
                          'has_classical', 'has_jazz', 'has_rock', 'has_pop', 'is_wind', 'is_rhythm', 'pca_tSNE']

        all_results = []

        for target_col in target_columns:
            col_name_str = "_and_".join(target_col) if isinstance(target_col, list) else str(target_col)
            self.logger.info(f"Computing RSA data for: {col_name_str}")
            records = []

            for i, clean_sub in enumerate(clean_ids):
                matched_rows = metadata_df[metadata_df['sub_id'] == clean_sub]
                if not matched_rows.empty:
                    if isinstance(target_col, list):
                        raw_vals = matched_rows[target_col].iloc[0]
                        if raw_vals.isna().any():
                            category = 'Unknown/None'
                        else:
                            category = " | ".join([f"{col}: {val}" for col, val in zip(target_col, raw_vals)])
                    else:
                        raw_val = matched_rows[target_col].values[0]
                        category = str(raw_val).strip()
                        if pd.isna(raw_val) or category.lower() in ['nan', 'none', '']:
                            category = 'Unknown/None'
                else:
                    category = 'Missing Data'

                records.append({'sub_id': clean_sub, 'category': category, 'signal': X[i]})

            df = pd.DataFrame(records)
            valid_mask = ~df['category'].isin(['Unknown/None', 'Missing Data'])
            df_valid = df[valid_mask].copy()
            df_valid = df_valid.sort_values(by=['category', 'sub_id']).reset_index(drop=True)

            unique_categories = df_valid['category'].unique()
            if len(unique_categories) < 2 or len(df_valid) < 5:
                continue

            X_valid = np.vstack(df_valid['signal'].values)
            labels = df_valid['category'].values

            brain_dist_vec = pdist(X_valid, metric='correlation')
            brain_rdm = squareform(brain_dist_vec)

            original_rho, global_p_value, permuted_rhos, model_rdm = permutation_test_rsa(brain_dist_vec, labels,
                                                                                          n_permutations=1000)

            boundaries = []
            unique_cats_ordered = []
            cat_starts = [0]
            current_cat = labels[0]
            unique_cats_ordered.append(current_cat)

            for idx, cat in enumerate(labels):
                if cat != current_cat:
                    boundaries.append(idx)
                    cat_starts.append(idx)
                    unique_cats_ordered.append(cat)
                    current_cat = cat
            cat_starts.append(len(labels))

            tick_positions = [(cat_starts[i] + cat_starts[i + 1]) / 2 for i in range(len(unique_cats_ordered))]

            n_cats = len(unique_cats_ordered)
            group_rdm = np.zeros((n_cats, n_cats))

            for i in range(n_cats):
                for j in range(n_cats):
                    idx_i = np.where(labels == unique_cats_ordered[i])[0]
                    idx_j = np.where(labels == unique_cats_ordered[j])[0]
                    sub_matrix = brain_rdm[np.ix_(idx_i, idx_j)]

                    if i == j:
                        if len(idx_i) > 1:
                            group_rdm[i, j] = np.mean(sub_matrix[np.triu_indices(len(idx_i), k=1)])
                        else:
                            group_rdm[i, j] = 0.0
                    else:
                        group_rdm[i, j] = np.mean(sub_matrix)

            cluster_1_names = []
            cluster_2_names = []

            if n_cats >= 2:
                group_rdm_sym = (group_rdm + group_rdm.T) / 2
                np.fill_diagonal(group_rdm_sym, 0)
                condensed_group_rdm = squareform(group_rdm_sym)
                Z = linkage(condensed_group_rdm, method='average')
                clusters = fcluster(Z, 2, criterion='maxclust')

                cluster_1_names = [unique_cats_ordered[i] for i in range(n_cats) if clusters[i] == 1]
                cluster_2_names = [unique_cats_ordered[i] for i in range(n_cats) if clusters[i] == 2]

            # Collect Pairwise AND Intra-group local p-values
            local_p_data = []
            n_permutations_local = 1000
            total_subjects = len(labels)

            for i in range(n_cats):
                for j in range(i, n_cats):
                    idx_i = np.where(labels == unique_cats_ordered[i])[0]
                    idx_j = np.where(labels == unique_cats_ordered[j])[0]

                    if i != j:
                        # Inter-group (Pairwise) -> Testing for SEPARATION (Distance > Chance)
                        combined_idx = np.concatenate([idx_i, idx_j])
                        n_i = len(idx_i)
                        true_dist = np.mean(brain_rdm[np.ix_(idx_i, idx_j)])
                        pair_brain_rdm = brain_rdm[np.ix_(combined_idx, combined_idx)]

                        count_extreme = 0
                        for _ in range(n_permutations_local):
                            shuff_idx = np.random.permutation(len(combined_idx))
                            shuff_dist = np.mean(pair_brain_rdm[np.ix_(shuff_idx[:n_i], shuff_idx[n_i:])])
                            # Notice the >= here: we want to know if the true distance is unusually LARGE
                            if shuff_dist >= true_dist: count_extreme += 1

                        p_val = (count_extreme + 1) / (n_permutations_local + 1)
                        local_p_data.append({'type': 'inter', 'i': i, 'j': j,
                                             'name': f"{unique_cats_ordered[i]} vs {unique_cats_ordered[j]}",
                                             'distance': true_dist,  # Added distance calculation
                                             'p': p_val})

                    else:
                        # Intra-group (Diagonal) -> Testing for COHESION (Distance < Chance)
                        n_i = len(idx_i)
                        if n_i > 1:
                            true_dist = group_rdm[i, i]
                            count_extreme = 0
                            for _ in range(n_permutations_local):
                                rand_subs = np.random.choice(total_subjects, size=n_i, replace=False)
                                sub_matrix = brain_rdm[np.ix_(rand_subs, rand_subs)]
                                shuff_dist = np.mean(sub_matrix[np.triu_indices(n_i, k=1)])
                                # Notice the <= here: we want to know if the true distance is unusually SMALL
                                if shuff_dist <= true_dist: count_extreme += 1

                            p_val = (count_extreme + 1) / (n_permutations_local + 1)
                            local_p_data.append({'type': 'intra', 'i': i, 'j': j,
                                                 'name': str(unique_cats_ordered[i]),
                                                 'distance': true_dist,  # Added distance calculation
                                                 'p': p_val})

            all_results.append({
                'col_name_str': col_name_str,
                'n_cats': n_cats,
                'unique_cats_ordered': unique_cats_ordered,
                'tick_positions': tick_positions,
                'boundaries': boundaries,
                'original_rho': original_rho,
                'global_p': global_p_value,
                'permuted_rhos': permuted_rhos,
                'model_rdm': model_rdm,
                'brain_rdm': brain_rdm,
                'group_rdm': group_rdm,
                'cluster_1_names': cluster_1_names,
                'cluster_2_names': cluster_2_names,
                'local_p_data': local_p_data
            })

        # =================================================================
        # Phase 2: Apply FDR Corrections (Global and Local)
        # =================================================================
        if all_results:
            # 1. Global FDR across all tested targets
            global_pvals = [res['global_p'] for res in all_results]
            global_qvals = fdr_bh(global_pvals)

            for res, q_val in zip(all_results, global_qvals):
                res['global_q'] = q_val

                # 2. Local FDR across pairs and intra-groups within this specific target
                local_pvals = [item['p'] for item in res['local_p_data']]
                local_qvals = fdr_bh(local_pvals)
                for item, local_q in zip(res['local_p_data'], local_qvals):
                    item['q'] = local_q

        # =================================================================
        # Phase 3: Plotting and Writing TXT
        # =================================================================
        for res in all_results:
            self.logger.info(f"Generating outputs for: {res['col_name_str']}")

            # Build matrices for Heatmap 5 (P-values and Q-values)
            p_val_matrix = np.full((res['n_cats'], res['n_cats']), np.nan)
            annot_matrix = np.full((res['n_cats'], res['n_cats']), "", dtype=object)

            for item in res['local_p_data']:
                i, j = item['i'], item['j']
                p, q = item['p'], item['q']
                p_val_matrix[i, j] = p_val_matrix[j, i] = p
                # Annotation format: p_orig \n (q_fdr)
                annot_text = f"{p:.3f}\n({q:.3f})"
                annot_matrix[i, j] = annot_matrix[j, i] = annot_text

            fig = plt.figure(figsize=(32, 8))
            gs = fig.add_gridspec(1, 5, width_ratios=[1, 1, 1.2, 1.2, 1.2])

            # 1. Categorical Model RDM
            ax1 = fig.add_subplot(gs[0])
            sns.heatmap(res['model_rdm'], cmap="Reds", xticklabels=False, yticklabels=False, ax=ax1,
                        cbar_kws={'label': 'Model Dissimilarity (1=Diff, 0=Same)'})
            ax1.set_title("Categorical Model RDM", fontsize=14, pad=10)
            ax1.set_yticks(res['tick_positions'])
            ax1.set_yticklabels(res['unique_cats_ordered'], rotation=0, fontsize=11, fontweight='bold')

            # 2. Brain Activity RDM
            ax2 = fig.add_subplot(gs[1])
            sns.heatmap(res['brain_rdm'], cmap="Blues", xticklabels=False, yticklabels=False, ax=ax2,
                        cbar_kws={'label': 'Brain Dissimilarity (Correlation Distance)'})
            ax2.set_title("Brain Activity RDM\n(Subject-Level)", fontsize=14, pad=10)
            ax2.set_yticks(res['tick_positions'])
            ax2.set_yticklabels(res['unique_cats_ordered'], rotation=0, fontsize=11, fontweight='bold')

            for b in res['boundaries']:
                ax1.axhline(b, color='white', linewidth=1)
                ax1.axvline(b, color='white', linewidth=1)
                ax2.axhline(b, color='black', linewidth=0.5, linestyle='--')
                ax2.axvline(b, color='black', linewidth=0.5, linestyle='--')

            # 3. Group-Level Mean Distance RDM
            ax3 = fig.add_subplot(gs[2])
            sns.heatmap(res['group_rdm'], cmap="Purples", annot=True, fmt=".3f",
                        xticklabels=res['unique_cats_ordered'], yticklabels=res['unique_cats_ordered'],
                        ax=ax3, cbar_kws={'label': 'Mean Distance'})
            ax3.set_title("Group-Level Proximity\n(Mean Brain Distance)", fontsize=14, pad=10)
            ax3.tick_params(axis='x', rotation=45)
            ax3.tick_params(axis='y', rotation=0, labelsize=11)

            # 4. Permutation Distribution
            ax4 = fig.add_subplot(gs[3])
            sns.histplot(res['permuted_rhos'], kde=True, ax=ax4, color='gray')
            label_txt = f"Observed Rho: {res['original_rho']:.3f}\np-val: {res['global_p']:.4f}\nq-val (FDR): {res['global_q']:.4f}"
            ax4.axvline(res['original_rho'], color='red', linestyle='--', linewidth=2, label=label_txt)
            ax4.set_title("Null Distribution\n(Global Permutation Test)", fontsize=14, pad=10)
            ax4.set_xlabel("Spearman Rho")
            ax4.set_ylabel("Frequency")
            ax4.legend(fontsize=12)

            # 5. Pairwise & Intra-group P-Values Heatmap (FDR annotated)
            ax5 = fig.add_subplot(gs[4])
            sns.heatmap(p_val_matrix, cmap="Reds_r", annot=annot_matrix, fmt="",
                        xticklabels=res['unique_cats_ordered'], yticklabels=res['unique_cats_ordered'],
                        ax=ax5, cbar_kws={'label': 'Permutation P-value'}, vmin=0, vmax=0.05)
            ax5.set_title("Local P-Values: p_orig \n(q_fdr) [Diag=Cohesion, Off=Separation]", fontsize=14, pad=10)
            ax5.tick_params(axis='x', rotation=45)
            ax5.tick_params(axis='y', rotation=0, labelsize=11)

            plt.suptitle(
                f"Representational Similarity Analysis (RSA)\nTarget: {res['col_name_str']} ({self.args.pc_str})",
                fontsize=18, fontweight='bold', y=1.08)
            plt.tight_layout()

            # Save the plot
            out_image_path = os.path.join(self.step_folder,
                                          f"rsa_analysis_{res['col_name_str']}_{self.args.pc_str}_{tm}.png")
            plt.savefig(out_image_path, dpi=300, bbox_inches='tight')
            plt.close()

            # =================================================================
            # Append Statistics to TXT (Clean, NO tables)
            # =================================================================
            with open(stats_out_path, 'a', encoding='utf-8') as f:
                f.write(f"{'=' * 60}\n")
                f.write(
                    f"Target: {res['col_name_str']:<30} | Rho: {res['original_rho']:>7.4f} | p-val: {res['global_p']:>7.4f} | q-val(FDR): {res['global_q']:>7.4f}\n\n")

                if res['n_cats'] >= 2:
                    f.write(f"--- Clustering Analysis (2 Main Clusters) ---\n")
                    f.write(f"Cluster 1: {', '.join(res['cluster_1_names'])}\n")
                    f.write(f"Cluster 2: {', '.join(res['cluster_2_names'])}\n\n")

                    inter_results = sorted([item for item in res['local_p_data'] if item['type'] == 'inter'],
                                           key=lambda x: x['p'])
                    intra_results = sorted([item for item in res['local_p_data'] if item['type'] == 'intra'],
                                           key=lambda x: x['p'])

                    if inter_results or intra_results:
                        f.write(f"--- Local Permutation Statistics (FDR Corrected) ---\n")

                    if inter_results:
                        best_inter = inter_results[0]
                        f.write(
                            f"Best Pair (Most Separated): {best_inter['name']} (p={best_inter['p']:.4f}, q={best_inter['q']:.4f})\n")

                    if intra_results:
                        f.write(f"Best Intra-Group (Most Cohesive): ")
                        best_intras = []
                        for intra in intra_results[:2]:
                            best_intras.append(f"{intra['name']} (p={intra['p']:.4f}, q={intra['q']:.4f})")
                        f.write(", ".join(best_intras) + "\n")
                f.write("\n")
        self.logger.info(f"RSA Analysis completed. Summary saved to: {stats_out_path}")

        # =================================================================
        # Export RSA Metrics to CSV for ML Script correlation
        # Includes inter-group distances, p/q values, and intra-group distances
        # =================================================================
        csv_path = os.path.join(self.step_folder, f"rsa_metrics_summary_{self.args.pc_str}_{tm}.csv")
        csv_records = []

        for res in all_results:
            # Create a dictionary of intra-group distances for quick lookup
            intra_dists = {
                item['name']: item.get('distance', np.nan)
                for item in res['local_p_data']
                if item['type'] == 'intra'
            }

            for item in res['local_p_data']:
                if item['type'] == 'inter':
                    # Split the pair name to identify the individual groups
                    parts = item['name'].split(' vs ')
                    group_a_name = parts[0].strip() if len(parts) > 0 else "Unknown"
                    group_b_name = parts[1].strip() if len(parts) > 1 else "Unknown"

                    csv_records.append({
                        'pair': item['name'],
                        'distance': item.get('distance', np.nan),
                        'rsa_p': item['p'],
                        'rsa_q': item.get('q', np.nan),
                        'group_a_name': group_a_name,
                        'group_a_intra': intra_dists.get(group_a_name, np.nan),
                        'group_b_name': group_b_name,
                        'group_b_intra': intra_dists.get(group_b_name, np.nan)
                    })

        if csv_records:
            df_csv = pd.DataFrame(csv_records)
            # Remove duplicates in case multiple iterations produced the same pairs
            df_csv = df_csv.drop_duplicates(subset=['pair'])
            df_csv.to_csv(csv_path, index=False)
            self.logger.info(f"Exported comprehensive pair distances to: {csv_path}")
