# -*- coding: utf-8 -*-
"""
Per-TR stimulus regressors from the bar-by-bar orchestration annotation workbook
(one sheet per movement), used by feature set 4 (encoding betas).

Workbook layout (header-less rows):
  row with label 'TR'           : scan TR (1-based) at which each bar starts
  row with label 'Timing (sec)' : bar start in seconds from the start of the music
  row with label 'Bars'         : bar number
  following rows                : one per instrument (column 0 = family in the
                                  workbook, column 1 = instrument) and the genre
                                  rows (column 0 = 'Genre'); values per bar:
                                  0 = not playing, 1 = playing, 2 = leading / solo
                                  (genre: 2 = dominant style)
Bars are the columns from index 3 on (column 2 holds the pre-music TR / timing).

Regressors (one value per TR, step functions over bars, then HRF-convolved and
z-scored) are built at the resolution of the pipeline's instrument groups:
  wind        = Woodwind + Brass rows of the workbook   (pipeline group 'Wind')
  guitars, keyboards, strings, drums                    (pipeline groups)
  jazz, rock, classic                                   genre levels
  density                                               number of instruments playing
With split_solo=True every instrument-group regressor becomes two columns,
<group>_playing (level >= 1) and <group>_solo (level == 2).
"""
import re
import numpy as np
import pandas as pd
from scipy.special import gammaln

# workbook family (column 0, lower case) -> pipeline instrument group (lower case)
WORKBOOK_FAMILY_TO_GROUP = {
    "woodwind": "wind", "brass": "wind",
    "guitars": "guitars", "keyboards": "keyboards", "strings": "strings", "drums": "drums",
}
GROUP_ORDER = ["wind", "guitars", "keyboards", "strings", "drums"]


def canonical_hrf(tr, duration=32.0):
    """SPM-style double-gamma HRF sampled at the TR."""
    t = np.arange(0, duration, tr)

    def gamma_pdf(x, shape, scale):
        return np.exp((shape - 1) * np.log(np.maximum(x, 1e-12)) - x / scale - gammaln(shape) - shape * np.log(scale))

    h = gamma_pdf(t, 6.0, 1.0) - gamma_pdf(t, 16.0, 1.0) / 6.0
    return h / h.sum()


def read_annotation_sheet(xlsx, sheet):
    """Returns (bar_start_trs, {(family, instrument): levels per bar})."""
    d = pd.read_excel(xlsx, sheet, header=None)
    labels = d.iloc[:, 1].astype(str).str.strip().str.lower()
    tr_rows = d.index[labels == "tr"]
    if len(tr_rows) == 0:
        raise ValueError(f"{xlsx} / sheet '{sheet}': no row labelled 'TR' in column B")
    tr_row = tr_rows[0]
    tr = pd.to_numeric(d.iloc[tr_row, 3:], errors="coerce").values
    ok = ~np.isnan(tr)
    tr = tr[ok]
    rows = {}
    for i in range(tr_row + 3, len(d)):
        fam = str(d.iloc[i, 0]).strip()
        name = str(d.iloc[i, 1]).strip()
        if fam.lower() in ("nan", "") or name.lower() in ("nan", "", "piece", "notes"):
            continue
        vals = pd.to_numeric(d.iloc[i, 3:], errors="coerce").values[ok]
        rows[(fam, name)] = np.nan_to_num(vals, nan=0.0)
    if not rows:
        raise ValueError(f"{xlsx} / sheet '{sheet}': no instrument rows found")
    return tr, rows


def step_to_trs(bar_tr, bar_values, n_total):
    """Hold each bar's value from its start TR (1-based) to the next bar start; silence after the last bar."""
    out = np.zeros(n_total)
    starts = np.round(np.asarray(bar_tr)).astype(int) - 1
    med = np.median(np.diff(starts)) if len(starts) > 1 else 1
    ends = np.append(starts[1:], starts[-1] + int(round(med)))
    for s, e, v in zip(starts, ends, bar_values):
        s, e = max(s, 0), min(e, n_total)
        if e > s:
            out[s:e] = v
    return out


def build_raw_regressors(tr_bars, rows, n_total, split_solo=False):
    inst = {k: v for k, v in rows.items() if k[0].lower() != "genre"}
    genre = {k: v for k, v in rows.items() if k[0].lower() == "genre"}
    cols = {}
    for group in GROUP_ORDER:
        members = [v for (f, _), v in inst.items() if WORKBOOK_FAMILY_TO_GROUP.get(f.lower()) == group]
        if not members:
            continue
        M = np.vstack(members)
        if split_solo:
            cols[f"{group}_playing"] = step_to_trs(tr_bars, (M >= 1).any(axis=0).astype(float), n_total)
            cols[f"{group}_solo"] = step_to_trs(tr_bars, (M == 2).any(axis=0).astype(float), n_total)
        else:
            cols[group] = step_to_trs(tr_bars, M.max(axis=0), n_total)
    unknown = sorted({f for (f, _) in inst if f.lower() not in WORKBOOK_FAMILY_TO_GROUP})
    if unknown:
        print(f"annotation: families not mapped to a pipeline group and ignored: {unknown}")
    for (_, g), v in genre.items():
        cols[re.sub(r"\\W+", "_", g.lower())] = step_to_trs(tr_bars, v, n_total)
    if inst:
        density = (np.vstack(list(inst.values())) >= 1).sum(axis=0).astype(float)
        cols["density"] = step_to_trs(tr_bars, density, n_total)
    return pd.DataFrame(cols)


def hrf_and_zscore(df, tr, logger=None):
    h = canonical_hrf(tr)
    out = {}
    for c in df.columns:
        x = df[c].values.astype(float)
        if x.std() == 0:
            # kept as zeros so that both movements have the same regressor set
            # (the feature layout of set 4 needs the same number per movement)
            msg = f"annotation: regressor '{c}' is constant in this movement; written as zeros"
            logger.warning(msg) if logger else print(msg)
            out[c] = np.zeros_like(x)
            continue
        conv = np.convolve(x, h)[:len(x)]
        out[c] = (conv - conv.mean()) / (conv.std() + 1e-12)
    return pd.DataFrame(out)


def build_annotation_regressors(xlsx, n_trs_mov1, n_trs_mov2, tr, n_skip_vols_start=0,
                                sheet_mov1="Movement 1", sheet_mov2="Movement 2",
                                split_solo=False, logger=None):
    """
    Returns {1: (regressors_df, raw_df), 2: (regressors_df, raw_df)}, each with
    exactly n_trs_mov<k> rows (the retained volumes) and the same columns.
    n_trs are the movement lengths of the fMRI data as loaded by the pipeline
    (after the skipped volumes); tr and n_skip_vols_start are the pipeline's.
    """
    result = {}
    for mov, sheet, n in [(1, sheet_mov1, n_trs_mov1), (2, sheet_mov2, n_trs_mov2)]:
        tr_bars, rows = read_annotation_sheet(xlsx, sheet)
        total = int(n) + int(n_skip_vols_start)
        last_bar_tr = int(np.round(tr_bars[-1]))
        if last_bar_tr > total:
            msg = (f"annotation: movement {mov}: the last bar starts at scan TR {last_bar_tr} but the fMRI data has "
                   f"only {total} volumes; bars beyond the data are cut")
            logger.warning(msg) if logger else print(msg)
        raw = build_raw_regressors(tr_bars, rows, total, split_solo=split_solo)
        reg = hrf_and_zscore(raw, tr, logger).iloc[n_skip_vols_start:].reset_index(drop=True)
        raw = raw.iloc[n_skip_vols_start:].reset_index(drop=True)
        result[mov] = (reg, raw)
        msg = (f"annotation: movement {mov}: {len(rows)} rows, {len(tr_bars)} bars (scan TR {int(tr_bars[0])}-{last_bar_tr}), "
               f"{reg.shape[0]} TRs x {reg.shape[1]} regressors: {list(reg.columns)}")
        logger.info(msg) if logger else print(msg)
    if list(result[1][0].columns) != list(result[2][0].columns):
        raise ValueError("annotation: the two movements produced different regressor sets")
    return result
