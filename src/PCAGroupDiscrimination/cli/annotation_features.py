#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone wrapper around PCAGroupDiscrimination.annotation_regressors: writes
acoustic_mov1.csv / acoustic_mov2.csv (and annotation_raw_mov<k>.csv) from the
orchestration annotation workbook. Inside the pipeline the same regressors are
built automatically from --annotation-xlsx, with the movement lengths, TR and
skipped volumes taken from the run; use this script only to inspect them.

Usage:
    python annotation_features.py --xlsx visualize.xlsx \
        --n-trs-mov1 <mov1_times> --n-trs-mov2 <mov2_times> --tr 0.75 --out-dir regressors/
"""
import os
import argparse
from PCAGroupDiscrimination.annotation_regressors import build_annotation_regressors


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--sheet-mov1", default="Movement 1")
    ap.add_argument("--sheet-mov2", default="Movement 2")
    ap.add_argument("--tr", type=float, default=0.75)
    ap.add_argument("--n-trs-mov1", type=int, required=True)
    ap.add_argument("--n-trs-mov2", type=int, required=True)
    ap.add_argument("--n-skip-vols-start", type=int, default=0)
    ap.add_argument("--split-solo", action="store_true")
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    res = build_annotation_regressors(a.xlsx, a.n_trs_mov1, a.n_trs_mov2, a.tr, a.n_skip_vols_start,
                                      a.sheet_mov1, a.sheet_mov2, a.split_solo)
    for mov, (reg, raw) in res.items():
        reg.to_csv(os.path.join(a.out_dir, f"acoustic_mov{mov}.csv"), index=False)
        raw.to_csv(os.path.join(a.out_dir, f"annotation_raw_mov{mov}.csv"), index=False)
    print(f"written to {a.out_dir}")


if __name__ == "__main__":
    main()
