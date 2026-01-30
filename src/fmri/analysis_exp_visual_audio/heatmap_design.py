import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

def add_correlation_to_heatmap(ax, df, pivot_df, title_base, use_va=False):
    """
    Add Pearson & Spearman correlations to each column of the heatmap
    """
    correlations = []
    if use_va:
        df_vis_prob = df[['Region', 'region_vis_prob_va']].drop_duplicates().set_index('Region').reindex(pivot_df.index)
    else:
        df_vis_prob = df[['Region', 'region_vis_prob']].drop_duplicates().set_index('Region').reindex(pivot_df.index)
    for i, pc_col in enumerate(pivot_df.columns):
        values = pivot_df[pc_col].values

        # Remove NaN pairs
        valid_mask = ~np.isnan(values)
        valid_values = values[valid_mask]
        if use_va:
            valid_probs = df_vis_prob['region_vis_prob_va'].values[valid_mask]
        else:
            valid_probs = df_vis_prob['region_vis_prob'].values[valid_mask]

        if len(valid_values) > 2:
            pearson_r, pearson_p = pearsonr(valid_values, valid_probs)
            spearman_r, spearman_p = spearmanr(valid_values, valid_probs)

            # Significance stars
            sig_p = "***" if pearson_p < 0.001 else "**" if pearson_p < 0.01 else "*" if pearson_p < 0.05 else ""
            sig_s = "***" if spearman_p < 0.001 else "**" if spearman_p < 0.01 else "*" if spearman_p < 0.05 else ""

            correlations.append({
                'pc': pc_col,
                'pearson_r': pearson_r,
                'spearman_r': spearman_r,
                'sig_p': sig_p,
                'sig_s': sig_s
            })

    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center', fontsize=9)

    # Create correlation text for x-axis label (below PC names)
    corr_text = "\n" + "\n".join(
        [f"{c['pc']}: r={c['pearson_r']:.2f}{c['sig_p']}, ρ={c['spearman_r']:.2f}{c['sig_s']}"
         for c in correlations])

    if use_va:
        ax.set_xlabel(f"Correlations with Visual/Auditory Prob:\n{corr_text}", fontsize=8)
    else:
        ax.set_xlabel(f"Correlations with Visual Prob:\n{corr_text}", fontsize=8)
    ax.set_title(title_base, fontsize=12, fontweight='bold')

    return correlations

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def color_labels(ax, region_vis_prob, use_va=False):
    for label in ax.get_yticklabels():
        label.set_fontsize(9)
        vis_prob = region_vis_prob[label.get_text()]

        if vis_prob > 0.5:
            strength = clamp(vis_prob)
            if use_va:
                label.set_color((0.55, 0.27, 0.07, strength))  # brown RGBA
            else:
                label.set_color((0.13, 0.55, 0.13, strength))  # green RGBA
        elif vis_prob < 0.5:
            strength = clamp(1 - vis_prob)
            if use_va:
                label.set_color((0.00, 0.60, 0.60, strength))  # turquoise RGBA
            else:
                label.set_color((0.50, 0.20, 0.60, strength))  # purple RGBA
        else:
            label.set_color((0.5, 0.5, 0.5, 0.6))  # tie → gray


def create_bold_annot_matrix(pivot_df, corr_col, pval_col, threshold=0.05):
    """
    Creates a string matrix for heatmap annotations with LaTeX bold formatting
    for statistically significant values.
    """
    # Select only the correlation part of the pivot for the base
    corr_df = pivot_df[corr_col]
    pval_df = pivot_df[pval_col]

    # Initialize the annotation matrix with string versions of correlations
    annot_matrix = corr_df.copy().astype(str)

    for region in corr_df.index:
        for pc in corr_df.columns:
            r_val = corr_df.loc[region, pc]
            p_val = pval_df.loc[region, pc]

            # Format the correlation coefficient to one decimal place
            formatted_r = f"{r_val:.1f}"

            # Apply LaTeX bold formatting if p-value is below the significance threshold
            if p_val < threshold:
                # \bf is the TeX command for bold; requires double braces for f-string and TeX
                annot_matrix.loc[region, pc] = f"$\\bf{{{formatted_r}}}$"
            else:
                annot_matrix.loc[region, pc] = formatted_r

    return annot_matrix
