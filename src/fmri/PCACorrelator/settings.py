import fnmatch

# =================================================================
# Brain Regions Mapping: Atlas labels to functional classifications
# =================================================================

def roi_vis_prob(atlas_labels, regions_class, feature):
    """
    Calculate normalized probability for regions given a feature.

    Parameters:
    -----------
    atlas_labels : list
        Atlas region labels
    regions_class : dict (optional)
        Pre-loaded regions classification dictionary.
    feature : str (optional)
        Feature to calculate probability for. If None, uses default behavior (backward compatible)

    Returns:
    --------
    region_vis_prob : dict
        Dictionary mapping region labels to normalized probabilities for the given feature
    """
    region_vis_prob = {}
    for label_bytes in atlas_labels:
        label = label_bytes.decode('utf-8')
        for pattern, info in regions_class.items():
            if fnmatch.fnmatch(label, pattern):
                prob = 0.5  # Default probability if no specific feature is given
                if feature is not None:
                    # Calculate probability for specific feature
                    # Get the weight for this feature
                    feature_weight = info.get(feature, 0.5)
                    # Get all feature weights and normalize
                    all_weights = {k: v for k, v in info.items() if k not in ['confidence']}
                    total_weight = sum(all_weights.values())
                    if total_weight > 0:
                        prob = feature_weight / total_weight
                    else:
                        prob = 0.5
                region_vis_prob[label] = prob
                break
        if label not in region_vis_prob:
            region_vis_prob[label] = 0.0
    return region_vis_prob
