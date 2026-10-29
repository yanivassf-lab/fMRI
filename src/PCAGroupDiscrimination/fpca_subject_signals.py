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
    """Reconstruct ROI deviations from the subject's spatial mean."""
    sub, dirs, n_pcs, min_len, atlas_img, atlas_labels = args

    if n_pcs < 1:
        raise ValueError("n_pcs must be positive")

    blocks = []

    for mov in (1, 2):
        rec = np.zeros((len(atlas_labels), min_len[mov]))

        for k in range(n_pcs):
            ts_file = os.path.join(
                dirs[mov],
                f"temporal_profile_pc_{k}.txt"
            )
            nii_file = os.path.join(
                dirs[mov],
                f"eigenfunction_{k}_importance_map.nii.gz"
            )

            # Require every requested component.
            if not (
                os.path.isfile(ts_file)
                and os.path.isfile(nii_file)
            ):
                raise FileNotFoundError(
                    f"Missing component: {sub}, movement {mov}, PC {k}"
                )

            # Section 7: average scores only inside the explicit brain mask.
            w = get_region_weights(
                nii_file, atlas_img, atlas_labels
            )
            t = load_temporal_profile(ts_file)

            # Reject incompatible lengths instead of silently truncating.
            if t.shape != (min_len[mov],):
                raise ValueError(
                    f"Time length mismatch: {sub}, movement {mov}, PC {k}"
                )

            rec += np.outer(w, t)

        # The scores represent deviations from the spatial mean.
        # Keep this representation: do not add mean_function.
        blocks.append(rec)

    return sub, np.concatenate(blocks, axis=1), n_pcs

def build_subject_fpca_signals(
    examples_dir, pattern, n_pcs, jobs=1, cache_path=None, logger=None
):
    """Build spatially centered ROI signals on matching acquisition times."""

    # Section 9: do not reuse unvalidated caches in the corrected analysis.
    cache_path = None

    logger = logger or logging.getLogger("clustering")

    if n_pcs < 1:
        raise ValueError("n_pcs must be positive")

    dataset = datasets.fetch_atlas_schaefer_2018(
        n_rois=100,
        yeo_networks=7,
        resolution_mm=2
    )
    atlas_img = nib.load(dataset.maps)
    atlas_labels = [
        label.decode("utf-8") if isinstance(label, bytes) else str(label)
        for label in dataset.labels
    ]

    subjects = _subject_dirs(examples_dir, pattern, logger)

    if not subjects:
        raise RuntimeError(
            f"No subjects with both movements under {examples_dir}"
        )

    # Require identical acquisition times within each movement.
    # Do not truncate all subjects to the shortest scan.
    min_len = {}

    for mov in (1, 2):
        reference = None

        for sub, dirs in subjects.items():
            archive = os.path.join(
                dirs[mov], "eigvecs_eigval_F.npz"
            )

            with np.load(archive, allow_pickle=False) as data:
                times = data["times"]

            if times.ndim != 1 or times.size == 0:
                raise ValueError(
                    f"Invalid acquisition times: {sub}, movement {mov}"
                )

            if reference is None:
                reference = times
            elif times.shape != reference.shape or not np.allclose(
                times, reference, rtol=0, atol=1e-10
            ):
                raise ValueError(
                    f"Different acquisition times: {sub}, movement {mov}"
                )

        min_len[mov] = len(reference)

    logger.info(
        "Centered subject-fPCA: %d subjects, %d PCs, "
        "movement lengths %d / %d",
        len(subjects), n_pcs, min_len[1], min_len[2]
    )

    args_list = [
        (sub, dirs, n_pcs, min_len, atlas_img, atlas_labels)
        for sub, dirs in subjects.items()
    ]

    if jobs and jobs > 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=jobs)(
            delayed(_reconstruct_subject)(args)
            for args in args_list
        )
    else:
        results = [
            _reconstruct_subject(args)
            for args in args_list
        ]

    data_dict = {
        sub: [matrix]
        for sub, matrix, _ in results
    }

    return data_dict, atlas_labels, min_len[1], min_len[2]
