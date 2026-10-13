# -*- coding: utf-8 -*-
"""
Subject-level fPCA reconstruction of the ROI time series, for feature set 4
(encoding betas) with --encoding-source fpca.

For every subject and movement the signal is rebuilt from the subject's OWN
functional principal components, produced by `fpca-main --mode pca-singles`:

    Y_hat[roi, t] = sum_k  w_k[roi] * t_k(t),   k = 0 .. n_pcs - 1

where t_k is the subject's temporal profile of component k
(temporal_profile_pc_<k>.txt in the subject folder) and w_k[roi] is the mean,
over the voxels of the ROI, of the subject's spatial scores of component k
(eigenfunction_<k>_importance_map.nii.gz in the subject folder, resampled to
the Schaefer-100 atlas).

Unlike the group reconstruction used by feature sets 0-2, which combines the
subject's spatial map with a GROUP temporal profile and therefore carries no
subject-specific dynamics, this reconstruction keeps the subject's own
temporal components: it is the B-spline-smoothed, GCV-penalised, rank-n_pcs
version of the subject's signal. The sign ambiguity of a component cancels in
the product w_k * t_k. The output has the format of the raw-data loader
({subject: [ROI x (mov1 + mov2) matrix]}, atlas labels, mov1_times, mov2_times),
so the encoding regression of feature set 4 runs on it unchanged. Train and
test subjects are handled alike (pca-singles writes the same files for both).
"""
import os
import re
import logging
import numpy as np
import joblib
import nibabel as nib
from nilearn import datasets

from .build_signals_from_files import get_region_weights, load_temporal_profile


def _subject_dirs(examples_dir, pattern, logger):
    group_dirs = {1: os.path.join(examples_dir, "outputs_mov1"), 2: os.path.join(examples_dir, "outputs_mov2")}
    subs = {1: {}, 2: {}}
    for mov in (1, 2):
        if not os.path.isdir(group_dirs[mov]):
            logger.warning(f"Movement directory not found: {group_dirs[mov]}")
            continue
        for d in os.listdir(group_dirs[mov]):
            path = os.path.join(group_dirs[mov], d)
            if os.path.isdir(path):
                m = re.search(pattern, d)
                if m:
                    subs[mov][m.group(1)] = path
    valid = sorted(set(subs[1]) & set(subs[2]))
    return {s: {1: subs[1][s], 2: subs[2][s]} for s in valid}


def _reconstruct_subject(args):
    sub, dirs, n_pcs, min_len, atlas_img, atlas_labels = args
    blocks = []
    for mov in (1, 2):
        rec = None
        n_used = 0
        for k in range(n_pcs):
            ts_file = os.path.join(dirs[mov], f"temporal_profile_pc_{k}.txt")
            nii_file = os.path.join(dirs[mov], f"eigenfunction_{k}_importance_map.nii.gz")
            if not (os.path.exists(ts_file) and os.path.exists(nii_file)):
                break
            w = get_region_weights(nii_file, atlas_img, atlas_labels)
            t = load_temporal_profile(ts_file)[:min_len[mov]]
            rec = np.outer(w, t) if rec is None else rec + np.outer(w, t)
            n_used += 1
        if rec is None:
            return sub, None, 0
        blocks.append(rec)
    return sub, np.concatenate(blocks, axis=1), n_used


def build_subject_fpca_signals(examples_dir, pattern, n_pcs, jobs=1, cache_path=None, logger=None):
    """
    Returns (data_dict, atlas_labels, mov1_times, mov2_times) with
    data_dict[sub] = [ROI x (mov1_times + mov2_times) reconstruction].
    """
    logger = logger or logging.getLogger("clustering")
    if cache_path and os.path.exists(cache_path):
        logger.info(f"Loading cached subject-fPCA reconstructions from: {cache_path}")
        data_dict, atlas_labels, mov1_times, mov2_times = joblib.load(cache_path)
        return data_dict, list(atlas_labels), mov1_times, mov2_times

    dataset = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
    atlas_img = nib.load(dataset.maps)
    atlas_labels = [l.decode("utf-8") if isinstance(l, bytes) else str(l) for l in dataset.labels]

    subjects = _subject_dirs(examples_dir, pattern, logger)
    if not subjects:
        raise RuntimeError(f"No subject folders found under {examples_dir}/outputs_mov{{1,2}} with pattern {pattern}")

    # common length per movement: the shortest subject-level profile
    min_len = {1: np.inf, 2: np.inf}
    for dirs in subjects.values():
        for mov in (1, 2):
            f = os.path.join(dirs[mov], "temporal_profile_pc_0.txt")
            if os.path.exists(f):
                min_len[mov] = min(min_len[mov], len(load_temporal_profile(f)))
    min_len = {m: int(v) for m, v in min_len.items()}
    logger.info(f"Subject-level fPCA reconstruction: {len(subjects)} subjects, {n_pcs} components, "
                f"movement lengths {min_len[1]} / {min_len[2]} TRs")

    args_list = [(s, d, n_pcs, min_len, atlas_img, atlas_labels) for s, d in subjects.items()]
    if jobs and jobs > 1:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=jobs)(delayed(_reconstruct_subject)(a) for a in args_list)
    else:
        results = [_reconstruct_subject(a) for a in args_list]

    data_dict = {}
    for sub, mat, n_used in results:
        if mat is None:
            logger.warning(f"{sub}: no subject-level components found, skipped")
            continue
        if n_used < n_pcs:
            logger.warning(f"{sub}: only {n_used} of {n_pcs} components found")
        data_dict[sub] = [mat]
    mov1_times, mov2_times = min_len[1], min_len[2]

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        joblib.dump((data_dict, atlas_labels, mov1_times, mov2_times), cache_path)
        logger.info(f"Subject-fPCA reconstructions cached to: {cache_path}")
    return data_dict, atlas_labels, mov1_times, mov2_times
