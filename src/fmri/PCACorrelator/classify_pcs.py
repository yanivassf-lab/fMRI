import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nibabel as nib
from nilearn.image import resample_to_img
from nilearn import datasets
import argparse
from scipy import stats
import csv

from settings import roi_vis_prob
from heatmap_design import color_labels, add_correlation_to_heatmap, create_bold_annot_matrix

# ==========================================
# CONFIGURATION
# ==========================================
# TR is now passed as a parameter to functions that need it


def load_blocks_from_csv(csv_path):
    """
    Load stimulus blocks from a CSV file.
    Expected columns: start_time, end_time, feature
    Returns a list of tuples: [(start, end, feature), ...]
    """
    blocks = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            start = float(row['start_time'])
            end = float(row['end_time'])
            feature = row['feature'].strip()
            blocks.append((start, end, feature))
    return blocks


def load_regions_class_from_csv(csv_path):
    """
    Load regions classification from a CSV file.
    Expected columns: region_pattern, and feature columns (v, a, va, x, etc.)
    Returns a dictionary similar to regions_class_claude_opus45 format

    CSV format:
    region_pattern,v,a,va,x,confidence,notes
    7Networks_*_Vis_*,0.95,0.05,0.5,0.5,0.95, ...
    ...

    Returns: {
        'region_pattern': {
            'feature1': weight1,
            'feature2': weight2,
            ...
            'confidence': conf_value
        },
        ...
    }
    """
    regions_class = {}
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            region_pattern = row['region_pattern'].strip()
            # Extract all feature columns (everything except region_pattern and confidence)
            features = {}
            confidence = float(row.get('confidence', 0.5))

            for key, value in row.items():
                if key not in ['region_pattern', 'confidence', 'notes']:
                    features[key] = float(value)

            regions_class[region_pattern] = {**features, 'confidence': confidence}

    return regions_class


def get_feature_color(feature, all_features=None):
    """
    Get color for a specific feature.

    Parameters:
    -----------
    feature : str
        Feature name
    all_features : list (optional)
        List of all features (used for auto-generating colors for unknown features)

    Returns:
    --------
    str
        Color name or hex code
    """

    # Auto-generate color for unknown features based on index
    if all_features is None:
        all_features = [feature]

    # Use a list of distinct colors for auto-assignment
    colors_list = [
        '#FF6B6B',  # Red
        '#4ECDC4',  # Teal
        '#45B7D1',  # Blue
        '#FFA07A',  # Light salmon
        '#98D8C8',  # Mint
        '#F7DC6F',  # Yellow
        '#BB8FCE',  # Purple
        '#85C1E2',  # Light blue
        '#F8B88B',  # Peach
        '#A9DFBF',  # Light green
        '#F5B7B1',  # Light pink
        '#D5A6BD',  # Mauve
        '#FAD7A0',  # Light orange
        '#AED6F1',  # Pale blue
        '#F1948A',  # Light red
        '#D2B4DE',  # Light purple
        '#A3E4D7',  # Light teal
        '#F9E79F',  # Light yellow
    ]

    try:
        feature_idx = list(all_features).index(feature)
    except ValueError:
        feature_idx = 0

    # Cycle through colors if more features than colors
    color = colors_list[feature_idx % len(colors_list)]

    return color


def get_feature_label(feature):
    """
    Get a display-friendly label for a feature.

    For known features, returns a nice label.
    For unknown features, just capitalizes the feature name.

    This is purely for display purposes and doesn't affect analysis.
    """
    # Replace underscores with spaces and capitalize
    return feature.replace('_', ' ').title()


def plot_temporal(temporal_sig, pc_name, peak_delay, output_path, tr, blocks):
    """
    Plot temporal profile with dynamic feature blocks based on blocks_design.csv
    Works with any feature names provided by the user.
    """
    plt.figure(figsize=(15, 5))
    times = np.arange(0, len(temporal_sig)) * tr
    plt.plot(times, temporal_sig, color='black', linewidth=1.5, label='Temporal')
    max_time = max(times)
    ticks = np.arange(0, max_time + 20, 20)
    plt.xticks(ticks, rotation=45)

    lag = int(peak_delay)

    # Get all unique features for color assignment
    all_features = sorted(set(cond for _, _, cond in blocks))
    legend_features = set()

    for s, e, cond in blocks:
        # Convert sec to TR indices
        s_idx = int(s) + lag
        e_idx = int(e) + lag

        # Get color for this feature (works with ANY feature name)
        color = get_feature_color(cond, all_features)
        label = get_feature_label(cond)

        # Add to legend only once per feature
        if cond not in legend_features:
            plt.axvspan(s_idx, e_idx, color=color, alpha=0.3, label=label)
            legend_features.add(cond)
        else:
            plt.axvspan(s_idx, e_idx, color=color, alpha=0.3)

    plt.title(f"(Temporal Trace for {pc_name})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Temporal Amplitude")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    output_file = os.path.join(output_path, f'temporal_{pc_name}.png')
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: {output_file}")


def spm_hrf(tr, peak_delay=6, duration=30):
    dt = tr
    t = np.arange(0, duration, dt)
    hrf = stats.gamma.pdf(t, peak_delay+1)
    return hrf / np.sum(hrf)


def calculate_regressor(n_points, blocks, tr, peak_delay, stimulus_duration_sec=None):
    """
    Creates regressors dynamically based on unique features in blocks.

    For each unique feature, creates:
    - A box function (stimulus timing)
    - A regressor (box convolved with HRF)

    Parameters:
    -----------
    n_points : int
        Number of timepoints
    blocks : list of tuples
        [(start_time, end_time, feature), ...]
    tr : float
        Repetition time
    peak_delay : float
        HRF peak delay
    stimulus_duration_sec : float or None
        Duration of stimulus (None for full block length)

    Returns:
    --------
    pd.DataFrame
        Columns: 'Model_<feature>' and 'box_<feature>' for each unique feature
    """
    # Identify unique features
    unique_features = sorted(set(cond for _, _, cond in blocks))

    timeline_len = n_points

    # Initialize box functions for all features
    boxes = {feat: np.zeros(timeline_len) for feat in unique_features}

    # Fill box functions based on blocks
    for s, e, cond in blocks:
        idx_s = int(s / tr)

        # Calculate effective end index
        if stimulus_duration_sec is not None:
            duration_idx = int(stimulus_duration_sec / tr)
            duration_idx = max(1, duration_idx)
            idx_e_eff = idx_s + duration_idx
        else:
            idx_e_eff = int(e / tr)

        # Prevent exceeding array length
        idx_e_eff = min(idx_e_eff, timeline_len)

        if idx_s >= timeline_len:
            continue

        boxes[cond][idx_s:idx_e_eff] = 1

    # Create regressors by convolving with HRF
    hrf = spm_hrf(tr, peak_delay)

    result_dict = {}
    for feature in unique_features:
        regressor = np.convolve(boxes[feature], hrf)[:timeline_len]
        result_dict[f'Model_{feature}'] = regressor
        result_dict[f'box_{feature}'] = boxes[feature]

    return pd.DataFrame(result_dict)


def calculate_corr(recon_sig, design_df):
    """
    Calculates correlations for all features in the design matrix.
    Returns dictionary of correlations and p-values for each feature.
    """
    # Extract unique features from design_df column names
    # Features are identified by 'box_<feature>' columns
    features = []
    for col in design_df.columns:
        if col.startswith('box_'):
            feature = col.replace('box_', '')
            features.append(feature)

    results = {}

    for feature in features:
        mask = design_df[f'box_{feature}'] > 0

        if np.sum(mask) > 2:  # Ensure enough points
            sig_sliced = recon_sig[mask]
            reg_sliced = design_df.loc[mask, f'Model_{feature}']
            r, p = stats.pearsonr(sig_sliced, reg_sliced, alternative='greater')
        else:
            r, p = 0, 1.0

        results[feature] = {'r': r, 'p': p}

    return results



def run_contrast_analysis(output_path, ts_files, nii_files, stimulus_duration_sec, peak_delay, tr, blocks, regions_class_dict):
    """
    Dynamic feature-based analysis. Creates heatmaps for each feature in blocks_design.csv
    """
    print("\n" + "=" * 60)
    print("DYNAMIC FEATURE-BASED ANALYSIS")
    print("=" * 60)

    # Load atlas
    dataset = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
    atlas_img = nib.load(dataset.maps)
    atlas_labels = dataset.labels

    # Get unique features from blocks
    unique_features = sorted(set(cond for _, _, cond in blocks))
    print(f"Features found: {unique_features}")

    # Get n_timepoints
    tmp = np.loadtxt(list(ts_files.values())[0])
    n_timepoints = tmp.shape[1] if tmp.ndim == 2 else len(tmp)

    ts_arr = np.loadtxt(ts_files['PC_0'])
    temporal_sig = ts_arr[1, :] if ts_arr.ndim == 2 else ts_arr
    n_points = len(temporal_sig)

    design_df = calculate_regressor(n_points, blocks, tr, peak_delay, stimulus_duration_sec)

    results = []
    for pc_name, ts_path in ts_files.items():
        if pc_name not in nii_files: continue
        print(f">> Processing {pc_name}...")

        # Load temporal profile
        ts_arr = np.loadtxt(ts_path)
        temporal_sig = ts_arr[1, :] if ts_arr.ndim == 2 else ts_arr
        plot_temporal(temporal_sig, pc_name, peak_delay, output_path, tr, blocks)

        n_points = len(temporal_sig)

        # Load spatial map
        pc_img = nib.load(nii_files[pc_name])
        resampled_atlas = resample_to_img(atlas_img, pc_img, interpolation='nearest',
                                          force_resample=True, copy_header=True)
        pc_data = pc_img.get_fdata()
        atlas_data = resampled_atlas.get_fdata()

        for i, label_bytes in enumerate(atlas_labels):
            roi_id = i + 1
            roi_name = label_bytes.decode('utf-8')
            roi_mask = (atlas_data == roi_id)

            if np.sum(roi_mask) == 0:
                continue

            spatial_weight = np.mean(pc_data[roi_mask])
            if abs(spatial_weight) < 0.0001:
                continue

            reconstructed_sig = temporal_sig * spatial_weight
            corr_results = calculate_corr(reconstructed_sig, design_df)
            # Create a result entry for each feature
            for feature in unique_features:
                r, p = corr_results[feature]['r'], corr_results[feature]['p']

                # Calculate region probability for this feature
                region_prob = roi_vis_prob(atlas_labels, regions_class_dict, feature)
                results.append({
                    'PC': pc_name,
                    'Region': roi_name,
                    'Feature': feature,
                    'Corr': r,
                    'Pval': p,
                    'region_vis_prob': region_prob[roi_name],
                })

    df = pd.DataFrame(results)

    # ==========================================
    # VISUALIZATION: CREATE HEATMAP FOR EACH FEATURE
    # ==========================================

    heatmap_data = {}

    for feature in unique_features:
        print(f"\n📊 Creating heatmap for feature: {feature}")

        # Filter data for this feature
        df_feature = df[df['Feature'] == feature].copy()

        if df_feature.empty:
            print(f"⚠️ No data for feature {feature}, skipping...")
            continue

        n_regions = len(df_feature['Region'].unique())
        fig_height = max(20, int(n_regions * 0.5))

        fig, ax = plt.subplots(1, 1, figsize=(20, fig_height))

        # Create pivot table for heatmap
        pivot_data = df_feature.pivot_table(
            index='Region',
            columns='PC',
            values='Corr',
            aggfunc='first'
        )
        # Create DataFrames for Corr and Pval, matching the heatmap's index/columns
        corr_df = df_feature.pivot_table(
            index='Region',
            columns='PC',
            values='Corr',
            aggfunc='first'
        )
        pval_df = df_feature.pivot_table(
            index='Region',
            columns='PC',
            values='Pval',
            aggfunc='first'
        )
        # Pass both as a DataFrame to create_bold_annot_matrix
        annot_matrix = create_bold_annot_matrix(
            pd.DataFrame({'Corr': corr_df.stack(), 'Pval': pval_df.stack()}).unstack(),
            'Corr',
            'Pval',
            threshold=0.05
        )

        # Create heatmap
        vmax = np.percentile(df_feature['Corr'].values, 97)
        vmin = np.percentile(df_feature['Corr'].values, 5)

        sns.heatmap(pivot_data, ax=ax, cmap='coolwarm', vmin=vmin, vmax=vmax,
                    annot=annot_matrix, fmt="", linewidths=0.5, annot_kws={'size': 8})

        # Get region probabilities for this feature
        region_prob_dict = roi_vis_prob(atlas_labels, regions_class_dict, feature)

        color_labels(ax, region_prob_dict)

        corr_result = add_correlation_to_heatmap(
            ax, df_feature, pivot_data,
            f'{get_feature_label(feature)} Correlations\n(Bold = p < 0.05)'
        )

        heatmap_data[feature] = corr_result

        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'heatmap_{feature}.png'), dpi=100, bbox_inches='tight')
        plt.close()

        print(f"✅ Saved: heatmap_{feature}.png")

    # Save results and analysis
    save_correlation_tables(heatmap_data, output_path)
    analyze_and_export_results(df, heatmap_data, output_path, correlation_threshold=0.2, top_n=10)

    # ==========================================
    # Find best PC per feature using pre-calculated correlations from heatmap_data
    # ==========================================
    feature_results = {}

    for feature in sorted(heatmap_data.keys()):
        # Get correlations already calculated in add_correlation_to_heatmap
        corr_list = heatmap_data[feature]  # List of dicts with 'pc', 'pearson_r', 'pearson_p', 'spearman_r', 'spearman_p'

        if not corr_list:
            continue

        # Find PC with best (highest absolute) pearson correlation
        best_pc_data = max(corr_list, key=lambda x: x['pearson_r'])
        best_pc_name = best_pc_data['pc']

        # Get all region correlations for the best PC
        best_pc_df = df[(df['Feature'] == feature) & (df['PC'] == best_pc_name)].copy()

        # Sort by absolute correlation
        best_pc_df = best_pc_df.sort_values('Corr', ascending=False)

        # Create region list with correlations
        region_data_list = []
        for _, row in best_pc_df.iterrows():
            region_data_list.append({
                'region': row['Region'],
                'correlation': float(row['Corr']),
                'p_value': float(row['Pval']),
                'region_probability': float(row['region_vis_prob'])
            })

        # Store results for this feature using pre-calculated correlations
        feature_results[feature] = {
            'best_pc': best_pc_name,
            'correlation_with_probabilities': float(best_pc_data['pearson_r']),
            'p_value': best_pc_data['sig_p'],  # Significance marker from heatmap_design
            'correlation_p_value': best_pc_data.get('pearson_p', np.nan),  # Actual p-value if available
            'spearman_r': float(best_pc_data['spearman_r']),
            'regions': region_data_list
        }

    # Save results to file
    output_file = os.path.join(output_path, 'best_pc_per_feature.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        for feature in sorted(feature_results.keys()):
            data = feature_results[feature]
            f.write(f"\n{'='*100}\n")
            f.write(f"FEATURE: {get_feature_label(feature).upper()}\n")
            f.write(f"{'='*100}\n")
            f.write(f"Best PC: {data['best_pc']}\n")
            f.write(f"Pearson correlation with region probabilities: {data['correlation_with_probabilities']:+.4f} {data['p_value']}\n")
            f.write(f"Spearman correlation: {data['spearman_r']:+.4f}\n")
            f.write(f"\n{'Region':<50} {'Correlation':>12} {'P-Value':>12} {'Region Prob':>12}\n")
            f.write(f"{'-'*86}\n")

            for i, region in enumerate(data['regions'], 1):
                f.write(f"{i:2d}. {region['region']:<48} {region['correlation']:>+12.4f} {region['p_value']:>12.6f} {region['region_probability']:>12.4f}\n")

    print(f"\n✅ Saved best PC results to: {output_file}")

    return feature_results

def save_correlation_tables(heatmap_data, output_path):
    """
    Saves and prints summary tables for all features.

    Parameters:
    -----------
    heatmap_data : dict
        Dictionary mapping feature -> correlation results list
    output_path : str
        Directory to save output
    """
    lines = []

    lines.append("=" * 80)
    lines.append("SPATIAL CORRELATION SUMMARY (ALL FEATURES)")
    lines.append("=" * 80)

    # Create a table for each feature
    for feature, corr_list in heatmap_data.items():
        lines.append(f"\n{'='*80}")
        lines.append(f"FEATURE: {get_feature_label(feature).upper()}")
        lines.append(f"{'='*80}")
        lines.append(f"{'PC':<10} | {'Pearson_r':<12} | {'Sig':<8} | {'Spearman_r':<12} | {'Sig':<8}")
        lines.append("-" * 80)

        for item in corr_list:
            pc = item.get('pc', 'Unknown')
            pearson_r = item.get('pearson_r', 0.0)
            sig_p = item.get('sig_p', 'n.s.')
            spearman_r = item.get('spearman_r', 0.0)
            sig_s = item.get('sig_s', 'n.s.')

            lines.append(f"{pc:<10} | {pearson_r:>+10.3f}  | {str(sig_p):<8} | {spearman_r:>+10.3f}  | {str(sig_s):<8}")

        lines.append("")

    # Print to screen
    print("\n")
    for line in lines:
        print(line)

    # Save to file
    output_file = os.path.join(output_path, 'spatial_correlation_summary.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\n✅ Saved Summary Table: {output_file}")


def analyze_and_export_results(df, heatmap_data, output_path, correlation_threshold=0.2, top_n=10):
    """
    Analyzes each PC to find the regions that drive it most strongly during each feature.
    """
    output_lines = []

    def log(text):
        print(text)
        output_lines.append(text)

    log("\n" + "=" * 80)
    log("DETAILED REGION ANALYSIS PER PC - ALL FEATURES")
    log("=" * 80)

    # Get unique features and PCs
    unique_features = sorted(df['Feature'].unique())
    pcs = sorted(df['PC'].unique())

    for pc_name in pcs:
        log(f"\n\n{'#' * 40} {pc_name} {'#' * 40}")

        for feature in unique_features:
            log(f"\n  {'='*70}")
            log(f"  FEATURE: {get_feature_label(feature).upper()}")
            log(f"  {'='*70}")

            # Filter data for this PC and feature
            df_pc_feat = df[(df['PC'] == pc_name) & (df['Feature'] == feature)].copy()

            if df_pc_feat.empty:
                log(f"  ⚠️ No data for this feature")
                continue

            log(f"  {'Region':<50} {'Corr':>10} {'Prob':>10}")
            log(f"  {'-' * 70}")

            # Sort by correlation descending
            top_regions = df_pc_feat.nlargest(top_n, 'Corr')
            for _, row in top_regions.iterrows():
                log(f"  {row['Region']:<50} {row['Corr']:+10.2f} {row['region_vis_prob']:10.2f}")

    # Export full dataframe to CSV
    csv_path = os.path.join(output_path, 'full_correlation_results.csv')
    df.to_csv(csv_path, index=False)
    log(f"\n\n💾 Full data saved to: {csv_path}")

    # Save the text report
    output_file = os.path.join(output_path, 'detailed_region_analysis.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(output_lines))

    print(f"✅ Saved Analysis Report: {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run dynamic feature-based fMRI analysis')
    parser.add_argument('--input-dir', '-p', required=True,
                        help='Path to fPCA output directory')
    parser.add_argument('--output-dir', '-o', required=True,
                        help='Path to output directory')
    parser.add_argument('--peak-delay', '-d', type=float, default=6.0, help='Measurment of the lag size in seconds')
    parser.add_argument('--n-pcs', '-n', type=int, default=7, help='Number of PCs to load')
    parser.add_argument('--stimulus-duration-sec', '-s', type=float, default=None,
                        help='Model of stimulus duration in seconds (None for full block)')
    parser.add_argument('--tr', type=float, default=0.75, help='Repetition time (TR) in seconds')
    parser.add_argument('--blocks-path', '-b', type=str, required=True,
                        help='Path to CSV file containing stimulus blocks (start_time, end_time, feature)')
    parser.add_argument('--regions-class-path', '-rc', type=str, required=True,
                        help='Path to CSV file containing regions classification with feature weights')
    args = parser.parse_args()

    ts_files = {f'PC_{i}': os.path.join(args.input_dir, f'temporal_profile_pc_{i}.txt') for i in range(args.n_pcs)}
    nii_files = {f'PC_{i}': os.path.join(args.input_dir, f'eigenfunction_{i}_importance_map.nii.gz') for i in
                 range(args.n_pcs)}
    os.makedirs(args.output_dir, exist_ok=True)

    # Load blocks from CSV file
    blocks = load_blocks_from_csv(args.blocks_path)

    # Load regions classification from CSV file
    regions_class_dict = load_regions_class_from_csv(args.regions_class_path)

    feature_results = run_contrast_analysis(args.output_dir, ts_files, nii_files, args.stimulus_duration_sec,
                                           args.peak_delay, args.tr, blocks, regions_class_dict)

    # Print summary of results
    print("\n" + "=" * 100)
    print("BEST PC PER FEATURE WITH REGION CORRELATIONS")
    print("=" * 100)

    for feature in sorted(feature_results.keys()):
        data = feature_results[feature]

        print(f"\n{'='*100}")
        print(f"FEATURE: {get_feature_label(feature).upper()}")
        print(f"{'='*100}")
        print(f"Best PC: {data['best_pc']}")

        # Handle p_value as either string (significance stars) or float
        p_val = data['p_value']
        p_str = p_val if isinstance(p_val, str) else f"{p_val:.6f}"
        print(f"Correlation with region probabilities: {data['correlation_with_probabilities']:>+.4f} ({p_str})")

        # Also show spearman if available
        if 'spearman_r' in data:
            print(f"Spearman correlation: {data['spearman_r']:>+.4f}")

        print(f"\n{'Region':<50} {'Correlation':>12} {'P-Value':>12} {'Region Prob':>12}")
        print(f"{'-'*86}")

        for i, region in enumerate(data['regions'][:15], 1):  # Show top 15
            print(f"{i:2d}. {region['region']:<48} {region['correlation']:>+12.4f} {region['p_value']:>12.6f} {region['region_probability']:>12.4f}")

"""
Example usage:
python classify_pcs.py \
  --input-dir /path/to/fmri \
  --output-dir /path/to/output \
  --blocks-path ./blocks_design.csv \
  --regions-class-path ./regions_class.csv \
  --tr 0.75 \
  --peak-delay 6.0
"""
