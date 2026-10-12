import os
import sys
import re
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, norm

from .roi_importance import summarize_roi_importance


class Statistics():
    def __init__(
        self,
        base_directory,
        align_dir,
        pc_num=None,
        extra_features_set=1,
        n_windows_per_exp=6,
        test_directory=None,
    ):
        """
        base_directory: the 2_ml_<pc> folder (training reports, saved models).
        test_directory: the 3_ml_test_<pc> folder (test reports). When given
            (mode 'stat-test'), the summary is written under the test folder and
            includes the test-set columns; otherwise (mode 'stat') it is written
            under the training folder without them.
        """
        self.base_directory = base_directory
        self.align_dir = align_dir
        self.test_directory = test_directory
        root = test_directory if test_directory else base_directory
        self.output_directory = os.path.join(root, "global_summary")
        self.pc_num = pc_num
        self.extra_features_set = extra_features_set
        self.n_windows_per_exp = n_windows_per_exp

    def parse_report_file(self, filepath):
        """
        Reads a single ML report text file and extracts key metrics using regex.
        Returns a dictionary with the extracted data.
        Note: ML p-value is parsed but no longer visualized as per request.
        """
        data = {
            "ml_model": None,
            "balanced_accuracy": None,
            "nm_prop_1": None,
            "perm_p": None,
            "perm_mean": None,
            "perm_std": None,
            "n_permutations": None,
            "perm_mode": None,
            "n_class_0": None,
            "n_class_1": None,
        }

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

            # Permutation test (training reports only)
            perm_match = re.search(r"Permutation p-value:\s+([0-9\.]+)", content)
            if perm_match:
                data["perm_p"] = float(perm_match.group(1))
            perm_mean_match = re.search(r"Permutation mean:\s+([0-9\.]+)", content)
            if perm_mean_match:
                data["perm_mean"] = float(perm_mean_match.group(1))
            perm_std_match = re.search(r"Permutation std:\s+([0-9\.]+)", content)
            if perm_std_match:
                data["perm_std"] = float(perm_std_match.group(1))
            n_perm_match = re.search(r"Permutations:\s+(\d+)", content)
            if n_perm_match:
                data["n_permutations"] = int(n_perm_match.group(1))
            mode_match = re.search(r"Permutation mode:\s+(\w+)", content)
            data["perm_mode"] = mode_match.group(1) if mode_match else "unknown"

            # Class sizes: the value_counts block printed after "Number of ... samples"
            # looks like "0    22" / "1     8" (class 0 = group_0, class 1 = group_1)
            for cls in ("0", "1"):
                cnt = re.search(rf"^\s*{cls}(?:\.0)?\s+(\d+)\s*$", content, flags=re.MULTILINE)
                if cnt:
                    data[f"n_class_{cls}"] = int(cnt.group(1))

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

    def compile_results(self, directory=None, folder_prefix="ml_", report_prefix="ml_report_"):
        """
        Walks through the directory structure, finds all relevant report files,
        and compiles them into a single pandas DataFrame.
        Used for the training reports (2_ml_<pc>/ml_<A>_vs_<B>/ml_report_*.txt)
        and, with other prefixes, for the test reports
        (3_ml_test_<pc>/ml_test_<A>_vs_<B>/test_report_*.txt).
        """
        results = []
        directory = directory or self.base_directory

        # Find all directories matching the pair pattern
        pair_dirs = glob.glob(os.path.join(directory, f"{folder_prefix}*_vs_*"))

        for pair_dir in pair_dirs:
            folder_name = os.path.basename(pair_dir)

            # Extract the groups from the folder name
            match = re.match(rf"{re.escape(folder_prefix)}(.+?)_vs_(.+)", folder_name)
            if not match:
                continue

            group_0, group_1 = match.groups()
            pair_name = f"{group_0} vs {group_1}"

            # Find all report txt files in this directory
            report_files = glob.glob(os.path.join(pair_dir, f"{report_prefix}*.txt"))

            for report_file in report_files:
                file_data = self.parse_report_file(report_file)

                # Only add if we successfully extracted the model name
                if file_data["ml_model"]:
                    file_data["pair"] = pair_name
                    file_data["group_0"] = group_0
                    file_data["group_1"] = group_1
                    file_data["report_dir"] = pair_dir
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

    def add_pair_level_columns(self, df, df_test=None):
        """
        Adds per-pair columns (repeated on every row of the pair):
          cv_mean_models, cv_max_models, cv_best_model, perm_p_best_model,
          n_train_0, n_train_1, rsa_rank (1 = largest distance among the pairs
          of this run), rsa_consistent ('yes' q<0.05, 'weak' p<0.05, 'rank'
          rank<=5, else 'no'),
        and, when test results are given (mode 'stat-test'):
          test_bacc (this row's model on the test set), test_mean_models,
          test_max_models, test_of_best_cv_model, min_of_means, n_test_0, n_test_1.
        The selection rule stays: the model is chosen by CV (cv_max_models);
        the test score of that model is test_of_best_cv_model.
        """
        if df.empty:
            return df

        g = df.groupby("pair")
        agg = pd.DataFrame({
            "cv_mean_models": g["balanced_accuracy"].mean().round(4),
            "cv_max_models": g["balanced_accuracy"].max().round(4),
        })
        best_idx = g["balanced_accuracy"].idxmax()
        agg["cv_best_model"] = df.loc[best_idx, "ml_model"].values
        agg["perm_p_best_model"] = df.loc[best_idx, "perm_p"].values
        agg["n_train_0"] = g["n_class_0"].first()
        agg["n_train_1"] = g["n_class_1"].first()

        # RSA rank among the pairs of this run (1 = most distant)
        dist = g["distance"].first()
        agg["rsa_rank"] = dist.rank(ascending=False, method="min")

        def consistency(pair):
            row = df[df["pair"] == pair].iloc[0]
            if pd.notna(row["rsa_q"]) and row["rsa_q"] < 0.05:
                return "yes"
            if pd.notna(row["rsa_p"]) and row["rsa_p"] < 0.05:
                return "weak"
            if pd.notna(agg.loc[pair, "rsa_rank"]) and agg.loc[pair, "rsa_rank"] <= 5:
                return "rank"
            return "no"
        agg["rsa_consistent"] = [consistency(p) for p in agg.index]

        if df_test is not None and not df_test.empty:
            t = df_test.rename(columns={"balanced_accuracy": "test_bacc",
                                        "n_class_0": "n_test_0", "n_class_1": "n_test_1"})
            df = df.merge(t[["pair", "ml_model", "test_bacc", "n_test_0", "n_test_1"]],
                          on=["pair", "ml_model"], how="left")
            gt = df.groupby("pair")
            agg["test_mean_models"] = gt["test_bacc"].mean().round(4)
            agg["test_max_models"] = gt["test_bacc"].max().round(4)
            # test score of the model selected by CV
            best = df.loc[best_idx, ["pair", "ml_model"]]
            agg["test_of_best_cv_model"] = [
                df[(df["pair"] == p) & (df["ml_model"] == m)]["test_bacc"].iloc[0]
                for p, m in zip(best["pair"], best["ml_model"])
            ]
            agg["min_of_means"] = agg[["cv_mean_models", "test_mean_models"]].min(axis=1)

        return df.merge(agg, left_on="pair", right_index=True, how="left")

    def write_roi_importance_for_selected_pairs(self, overview, min_test=0.60, max_perm_p=0.05):
        """
        stat-test only. The ROI importance tables were computed by mode 'stat'
        from the saved models (they do not depend on the test set), so they are
        not recomputed; here they are filtered to the pairs that passed the test:
            test_of_best_cv_model >= min_test  and  perm_p_best_model <= max_perm_p,
        keeping only the rows of the CV-selected model of each pair.
        Written to roi_importance_selected_pairs.csv next to the summary (plus
        the per-movement files when they exist).
        """
        src_dir = os.path.join(self.base_directory, "global_summary")
        needed = {"test_of_best_cv_model", "best_model_p", "cv_best_model"}
        if not needed.issubset(overview.columns):
            return
        sel = overview[(overview["test_of_best_cv_model"] >= min_test)
                       & (overview["best_model_p"] <= max_perm_p)]
        source = overview["best_model_p_source"].iloc[0] if "best_model_p_source" in overview.columns else "permutation"
        criteria = f"test_of_best_cv_model >= {min_test} and best_model_p ({source}) <= {max_perm_p}"
        if sel.empty:
            print(f"ROI importance: no pair satisfies {criteria}; nothing written.")
            return
        keep = set(zip(sel["pair"], sel["cv_best_model"]))

        for src_name, out_name in [("roi_importance_all.csv", "roi_importance_selected_pairs.csv"),
                                   ("roi_importance_mov1_all.csv", "roi_importance_selected_pairs_mov1.csv"),
                                   ("roi_importance_mov2_all.csv", "roi_importance_selected_pairs_mov2.csv")]:
            src = os.path.join(src_dir, src_name)
            if not os.path.exists(src):
                if src_name == "roi_importance_all.csv":
                    print(f"ROI importance: {src} not found; run mode 'stat' first.")
                continue
            roi = pd.read_csv(src)
            roi = roi[[(p, m) in keep for p, m in zip(roi["pair"], roi["ml_model"])]]
            roi = roi.merge(sel[["pair", "cv_best_model", "cv_max_models", "test_of_best_cv_model", "best_model_p"]],
                            on="pair", how="left")
            roi = roi.sort_values(["pair", "rank"])
            out = os.path.join(self.output_directory, out_name)
            roi.to_csv(out, index=False)
            print(f"ROI importance for {len(keep)} selected pair(s) ({criteria}) saved to: {out}")

    # ------------------------------------------------------------------
    # Significance
    # ------------------------------------------------------------------
    @staticmethod
    def _bh_fdr(pvals):
        """Benjamini-Hochberg adjusted p-values (NaN-safe)."""
        p = np.asarray(pvals, dtype=float)
        q = np.full_like(p, np.nan)
        ok = ~np.isnan(p)
        if ok.sum() == 0:
            return q
        pv = p[ok]
        n = len(pv)
        order = np.argsort(pv)
        ranked = pv[order] * n / (np.arange(n) + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(n)
        out[order] = np.minimum(ranked, 1.0)
        q[ok] = out
        return q

    @staticmethod
    def _null_sd_bacc(n0, n1):
        """SD of a balanced accuracy at chance level for class sizes n0, n1."""
        try:
            return 0.25 * np.sqrt(1.0 / float(n0) + 1.0 / float(n1))
        except (TypeError, ZeroDivisionError, ValueError):
            return np.nan

    MIN_PERMUTATIONS_FOR_PERM_TEST = 100

    def ensemble_from_cv_results(self, pair_dir):
        """
        Majority vote of the out-of-fold CV predictions of all models of a pair
        (ml_cv_results_<model>_*.csv: sub_id, y_classical_tendency, cv_pred [, cv_prob]).
        Returns (ensemble_bacc, n_models, n_subjects_all_models_correct, n_subjects)
        or (nan, 0, nan, nan) when the files are missing. One subject counted
        once per model (the latest file per model is used).
        """
        files = glob.glob(os.path.join(pair_dir, "ml_cv_results_*.csv"))
        latest = {}
        for f in files:
            m = re.search(r"ml_cv_results_([A-Za-z]+)_", os.path.basename(f))
            if m:
                model = m.group(1)
                if model not in latest or os.path.getmtime(f) > os.path.getmtime(latest[model]):
                    latest[model] = f
        preds = []
        for model, f in latest.items():
            try:
                d = pd.read_csv(f)
                col = "cv_prob" if "cv_prob" in d.columns and d["cv_prob"].notna().all() else "cv_pred"
                preds.append(d.set_index("sub_id")[[col, "y_classical_tendency"]].rename(columns={col: model}))
            except Exception:
                continue
        if not preds:
            return np.nan, 0, np.nan, np.nan
        joined = preds[0]
        for d in preds[1:]:
            joined = joined.join(d.drop(columns="y_classical_tendency"), how="inner")
        models = [c for c in joined.columns if c != "y_classical_tendency"]
        y = joined["y_classical_tendency"].astype(int).values
        score = joined[models].astype(float).mean(axis=1).values          # mean prob (or mean of 0/1 votes)
        vote = (score > 0.5).astype(int)
        correct_all = (joined[models].astype(float).gt(0.5).astype(int).eq(y[:, None])).all(axis=1).sum()
        tpr = (vote[y == 1] == 1).mean() if (y == 1).any() else np.nan
        tnr = (vote[y == 0] == 0).mean() if (y == 0).any() else np.nan
        return float(np.nanmean([tpr, tnr])), len(models), int(correct_all), int(len(y))

    def add_significance_columns(self, df, threshold=0.60):
        """
        Adds significance columns and returns (df, summary dict).

        Per (pair, model) row
          perm_q           BH-FDR of the permutation p-value over all rows of the run.
        Per pair (repeated on its rows)
          row_mean_z/p/q   test of the MEAN balanced accuracy over the models of the
                           pair against chance. The models of one pair are fitted on
                           the same subjects, so their scores are correlated and the
                           mean does not have the variance of an average of
                           independent scores. We therefore use the CONSERVATIVE
                           null SD of a single model (perfect correlation):
                             z = (mean_models - null_mean) / null_sd_single,
                           with null_mean and null_sd_single the averages of the
                           permutation mean and std of the models (fallback: 0.5 and
                           the analytic SD for the class sizes). p is one-sided.
          row_mean_p_indep the optimistic p-value assuming independent models
                           (SD / sqrt(M)); the truth lies between the two.
          test_sel_z/p/q   (stat-test) test score of the CV-selected model vs chance,
                           analytic null SD for the test class sizes, one-sided.
          test_row_mean_p  (stat-test) same conservative test for the mean test
                           score over the models.
        Summary (count tests)
          how many (pair, model) scores and how many pair means exceed `threshold`,
          versus the number expected under the null; the p-value for the count
          assumes independence between combinations and is therefore optimistic.
        """
        summary = {"threshold": threshold}
        if df.empty:
            return df, summary

        # ---- per row: FDR over permutation p-values
        df["perm_q"] = self._bh_fdr(df["perm_p"].values)

        # ---- per pair: row-mean test (CV)
        # permutation-based quantities are used only when enough permutations were run
        n_perm = df["n_permutations"].dropna()
        modes = set(df["perm_mode"].dropna()) if "perm_mode" in df.columns else set()
        nested = bool(modes) and modes <= {"nested"}
        perm_ok = (not n_perm.empty) and (n_perm.min() >= self.MIN_PERMUTATIONS_FOR_PERM_TEST) and nested
        summary["n_permutations"] = int(n_perm.min()) if not n_perm.empty else None
        summary["permutation_mode"] = ",".join(sorted(modes)) if modes else "unknown"
        summary["permutation_test_used"] = bool(perm_ok)
        if not perm_ok:
            df["perm_q"] = np.nan   # p-values from a handful of permutations carry no information

        # analytic p per (pair, model) row and its FDR over all rows of the run
        sd_rows_all = np.array([self._null_sd_bacc(a, b) for a, b in zip(df["n_class_0"], df["n_class_1"])])
        with np.errstate(invalid="ignore", divide="ignore"):
            df["analytic_p"] = norm.sf((df["balanced_accuracy"].values - 0.5) / sd_rows_all)
        df["analytic_q"] = self._bh_fdr(df["analytic_p"].values)

        g = df.groupby("pair")
        rows = []
        for pair, sub in g:
            M = int(sub["balanced_accuracy"].notna().sum())
            mean_obs = sub["balanced_accuracy"].mean()
            sd_an = self._null_sd_bacc(sub["n_class_0"].iloc[0], sub["n_class_1"].iloc[0])
            z_an = (mean_obs - 0.5) / sd_an if pd.notna(sd_an) and sd_an > 0 else np.nan
            if perm_ok and sub["perm_std"].notna().any():
                null_mean = sub["perm_mean"].mean(); null_sd = sub["perm_std"].mean()
                z = (mean_obs - null_mean) / null_sd if null_sd and null_sd > 0 else np.nan
            else:
                null_mean, null_sd, z = 0.5, sd_an, z_an
            ens_bacc, ens_n, ens_all_correct, ens_n_subj = self.ensemble_from_cv_results(sub["report_dir"].iloc[0]) \
                if "report_dir" in sub.columns else (np.nan, 0, np.nan, np.nan)
            z_ens = (ens_bacc - 0.5) / sd_an if pd.notna(ens_bacc) and pd.notna(sd_an) and sd_an > 0 else np.nan
            rows.append({
                "pair": pair,
                "row_n_models": M,
                "row_null_mean": null_mean,
                "row_null_sd_single": null_sd,
                "row_mean_z": z,
                "row_mean_p": norm.sf(z) if pd.notna(z) else np.nan,
                "row_mean_p_analytic": norm.sf(z_an) if pd.notna(z_an) else np.nan,
                "row_mean_p_indep": norm.sf(z * np.sqrt(M)) if pd.notna(z) and M > 0 else np.nan,
                "n_models_above_chance": int((sub["balanced_accuracy"] > 0.5).sum()),
                "n_models_above_1sd": int((sub["balanced_accuracy"] > 0.5 + sd_an).sum()) if pd.notna(sd_an) else np.nan,
                "ensemble_bacc": ens_bacc,
                "ensemble_p": norm.sf(z_ens) if pd.notna(z_ens) else np.nan,
                "ensemble_n_models": ens_n,
                "subjects_correct_by_all_models": ens_all_correct,
                "n_subjects_cv": ens_n_subj,
            })
        P = pd.DataFrame(rows).set_index("pair")
        P["row_mean_q"] = self._bh_fdr(P["row_mean_p"].values)
        P["ensemble_q"] = self._bh_fdr(P["ensemble_p"].values)
        df = df.merge(P, left_on="pair", right_index=True, how="left")

        # p-value of the CV-selected model: permutation-based when usable, else analytic
        best_rows = df.loc[df.groupby("pair")["balanced_accuracy"].idxmax()]
        src_p, src_q = ("perm_p", "perm_q") if perm_ok else ("analytic_p", "analytic_q")
        best_p = best_rows.set_index("pair")[src_p]; best_q = best_rows.set_index("pair")[src_q]
        df["best_model_p"] = df["pair"].map(best_p)
        df["best_model_q"] = df["pair"].map(best_q)
        df["best_model_p_source"] = "permutation" if perm_ok else "analytic"

        # ---- count tests (CV)
        sd_rows = np.array([self._null_sd_bacc(a, b) for a, b in zip(df["n_class_0"], df["n_class_1"])])
        summary.update(self._count_test(
            scores=df["balanced_accuracy"].values,
            null_means=df["perm_mean"].fillna(0.5).values if perm_ok else np.full(len(df), 0.5),
            null_sds=df["perm_std"].fillna(pd.Series(sd_rows)).values if perm_ok else sd_rows,
            threshold=threshold, prefix="cv_rows"))
        summary.update(self._count_test(
            scores=P.index.map(g["balanced_accuracy"].mean()).values,
            null_means=P["row_null_mean"].values, null_sds=P["row_null_sd_single"].values,
            threshold=threshold, prefix="cv_pair_means"))

        # ---- test set (stat-test)
        if "test_bacc" in df.columns:
            rows = []
            for pair, sub in df.groupby("pair"):
                n0, n1 = sub["n_test_0"].iloc[0], sub["n_test_1"].iloc[0]
                sd = self._null_sd_bacc(n0, n1)
                sel = sub["test_of_best_cv_model"].iloc[0]
                z_sel = (sel - 0.5) / sd if pd.notna(sel) and pd.notna(sd) and sd > 0 else np.nan
                mean_t = sub["test_bacc"].mean()
                z_row = (mean_t - 0.5) / sd if pd.notna(mean_t) and pd.notna(sd) and sd > 0 else np.nan
                rows.append({"pair": pair, "test_null_sd": sd,
                             "test_sel_z": z_sel, "test_sel_p": norm.sf(z_sel) if pd.notna(z_sel) else np.nan,
                             "test_row_mean_z": z_row, "test_row_mean_p": norm.sf(z_row) if pd.notna(z_row) else np.nan})
            T = pd.DataFrame(rows).set_index("pair")
            T["test_sel_q"] = self._bh_fdr(T["test_sel_p"].values)
            T["test_row_mean_q"] = self._bh_fdr(T["test_row_mean_p"].values)
            df = df.merge(T, left_on="pair", right_index=True, how="left")
            sds = np.array([self._null_sd_bacc(a, b) for a, b in zip(df["n_test_0"], df["n_test_1"])])
            summary.update(self._count_test(scores=df["test_bacc"].values, null_means=np.full(len(df), 0.5),
                                            null_sds=sds, threshold=threshold, prefix="test_rows"))
            per_pair = df.drop_duplicates("pair")
            summary.update(self._count_test(scores=per_pair["test_of_best_cv_model"].values,
                                            null_means=np.full(len(per_pair), 0.5),
                                            null_sds=per_pair["test_null_sd"].values,
                                            threshold=threshold, prefix="test_selected"))
        return df, summary

    @staticmethod
    def _count_test(scores, null_means, null_sds, threshold, prefix):
        """
        Observed number of scores >= threshold versus the number expected under
        the null (normal approximation per score), with a Poisson-binomial
        normal-approximation p-value that ASSUMES INDEPENDENCE between scores
        (optimistic, since models of one pair share the subjects).
        """
        scores = np.asarray(scores, dtype=float); mu = np.asarray(null_means, dtype=float); sd = np.asarray(null_sds, dtype=float)
        ok = ~np.isnan(scores) & ~np.isnan(mu) & ~np.isnan(sd) & (sd > 0)
        if ok.sum() == 0:
            return {}
        p_each = norm.sf((threshold - mu[ok]) / sd[ok])
        observed = int((scores[ok] >= threshold).sum())
        expected = float(p_each.sum())
        var = float((p_each * (1 - p_each)).sum())
        p_count = norm.sf((observed - expected) / np.sqrt(var)) if var > 0 else np.nan
        return {f"{prefix}_n": int(ok.sum()), f"{prefix}_observed_ge_thr": observed,
                f"{prefix}_expected_ge_thr": expected, f"{prefix}_count_p_indep": p_count}

    def write_significance_summary(self, df, summary):
        """Writes significance_summary.txt next to the CSV files."""
        thr = summary.get("threshold", 0.60)
        path = os.path.join(self.output_directory, "significance_summary.txt")
        n_perm = summary.get("n_permutations")
        perm_mode = summary.get("permutation_mode", "unknown")
        perm_note = (f"Permutation tests used ({n_perm} nested permutations)." if summary.get("permutation_test_used")
                     else f"Permutation tests NOT used ({n_perm} permutations, mode '{perm_mode}'; required: >= "
                          f"{self.MIN_PERMUTATIONS_FOR_PERM_TEST} nested permutations, see --run-permutations): "
                          "all p-values below are analytic, from the null SD of a balanced accuracy for the class sizes, "
                          "0.25*sqrt(1/n0 + 1/n1).")
        lines = ["=== Significance summary ===", "", perm_note, "",
                 "Per (pair, model): analytic p-value (analytic_p) and, when usable, the permutation p-value",
                 "(perm_p) of the training report, each with BH-FDR over all rows of this run (analytic_q, perm_q).",
                 "The p and q printed after 'best' are those of the CV-selected model; its q accounts for all",
                 "rows of the run, which is the correction needed when the best of several models is reported.", "",
                 "Per pair, mean over models (columns row_mean_z/p/q): the models of a pair are fitted on",
                 "the same subjects, so their scores are correlated. The reported p is CONSERVATIVE",
                 "(null SD of a single model, i.e. perfect correlation); row_mean_p_indep is the",
                 "optimistic bound assuming independent models. The truth lies in between.", "",
                 "Agreement between models: n_models_above_chance / n_models_above_1sd count the models of the pair",
                 "above 0.5 / above 0.5 + one null SD; ensemble_bacc is the balanced accuracy of the majority vote",
                 "of the models' out-of-fold predictions (one statistic for 'all models see the same difference'),",
                 "with its analytic p and FDR q; subjects_correct_by_all_models counts subjects every model got right.",
                 "Agreement shows the difference is in the data rather than in a model's randomness; it does not",
                 "add evidence against sampling noise, because all models see the same subjects.", ""]

        def block(prefix, label):
            if f"{prefix}_n" not in summary:
                return []
            return [f"{label}: {summary[f'{prefix}_observed_ge_thr']} of {summary[f'{prefix}_n']} scores >= {thr:.2f}; "
                    f"expected under the null {summary[f'{prefix}_expected_ge_thr']:.1f}; "
                    f"p (independence assumed, optimistic) = {summary[f'{prefix}_count_p_indep']:.3g}"]

        lines += ["--- Counts above the threshold ---"]
        lines += block("cv_rows", "CV, all (pair, model) combinations")
        lines += block("cv_pair_means", "CV, pair means over models")
        lines += block("test_rows", "Test, all (pair, model) combinations")
        lines += block("test_selected", "Test, CV-selected model per pair")
        lines += ["", "--- Pairs ---"]
        per_pair = df.drop_duplicates("pair").sort_values("row_mean_p")
        for _, r in per_pair.iterrows():
            line = (f"{r['pair']:<24} mean CV {r['cv_mean_models']:.3f}  z {r['row_mean_z']:+.2f}  "
                    f"p {r['row_mean_p']:.3g} (q {r['row_mean_q']:.3g}; analytic {r['row_mean_p_analytic']:.3g}; indep {r['row_mean_p_indep']:.3g})"
                    f"  models>0.5 {int(r['n_models_above_chance'])}/{int(r['row_n_models'])}"
                    f"  ensemble {r['ensemble_bacc']:.3f} p {r['ensemble_p']:.3g}"
                    f"  best {r['cv_best_model']} ({r['cv_max_models']:.3f}, p {r['best_model_p']:.3g}, q {r['best_model_q']:.3g} {r['best_model_p_source']})")
            if "test_sel_p" in r and pd.notna(r.get("test_sel_p", np.nan)):
                line += (f"  | test of best {r['test_of_best_cv_model']:.3f} p {r['test_sel_p']:.3g} (q {r['test_sel_q']:.3g})"
                         f"  test mean {r['test_mean_models']:.3f} p {r['test_row_mean_p']:.3g}")
            lines.append(line)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Significance summary saved to: {path}")

    def write_pairs_overview(self, df):
        """
        One row per pair, the per-pair columns only, sorted by min_of_means
        (stat-test) or by cv_mean_models (stat). Same information as
        all_pairs_summary.csv, in the compact form used for reporting.
        """
        cols = ["pair", "n_train_0", "n_train_1", "n_test_0", "n_test_1",
                "cv_mean_models", "cv_max_models", "cv_best_model", "perm_p_best_model", "best_model_p", "best_model_q", "best_model_p_source",
                "row_mean_z", "row_mean_p", "row_mean_q", "row_mean_p_analytic", "row_mean_p_indep",
                "n_models_above_chance", "n_models_above_1sd", "ensemble_bacc", "ensemble_p", "ensemble_q",
                "subjects_correct_by_all_models", "n_subjects_cv",
                "test_mean_models", "test_max_models", "test_of_best_cv_model", "min_of_means",
                "test_sel_p", "test_sel_q", "test_row_mean_p", "test_row_mean_q",
                "distance", "rsa_rank", "rsa_p", "rsa_q", "rsa_consistent"]
        cols = [c for c in cols if c in df.columns]
        overview = df.drop_duplicates("pair")[cols]
        sort_col = "min_of_means" if "min_of_means" in overview.columns else "cv_mean_models"
        overview = overview.sort_values(sort_col, ascending=False)
        path = os.path.join(self.output_directory, "pairs_overview.csv")
        overview.to_csv(path, index=False)
        print(f"Per-pair overview saved to: {path}")
        return overview

    def generate_visualizations(self, df, value_col="balanced_accuracy",
                                title="Balanced Accuracy by Pair and Model",
                                filename="summary_balanced_accuracy.png"):
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
            values=value_col,
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

            top_3_bacc_pairs = set(model_df.nlargest(3, value_col)['pair'])

            overlap_score = len(top_3_dist_pairs.intersection(top_3_bacc_pairs))
            top_3_scores[col] = overlap_score

            valid_data = model_df[[value_col, 'distance']].dropna()
            if len(valid_data) > 1 and valid_data['distance'].nunique() > 1:
                corr, _ = pearsonr(valid_data['distance'], valid_data[value_col])
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
        ax1.set_xticklabels(x_labels_bacc, rotation=0, fontsize=11)  # slightly reduced font size to fit three lines

        plt.title(title, fontsize=18, pad=20)
        plt.ylabel("Instrument Pair (with RSA Metrics)", fontsize=14, labelpad=15)
        plt.xlabel("ML Model", fontsize=14, labelpad=15)

        plt.subplots_adjust(left=0.35, bottom=0.18, top=0.95)  # increase bottom margin to accommodate the new score
        plt.savefig(os.path.join(self.output_directory, filename), dpi=300, bbox_inches='tight')
        plt.close()

        txt_path = os.path.join(self.output_directory, filename.replace(".png", "_correlations.txt")
                                if filename != "summary_balanced_accuracy.png" else "correlations_summary.txt")
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

        # Test-set reports (mode 'stat-test' only)
        df_test = None
        if self.test_directory:
            df_test = self.compile_results(self.test_directory, folder_prefix="ml_test_",
                                           report_prefix="test_report_")
            if df_test.empty:
                print(f"Notice: no test reports found under {self.test_directory}; test columns will be empty.")
                df_test = None

        df_results = self.add_pair_level_columns(df_results, df_test)
        df_results, significance = self.add_significance_columns(df_results)

        # Reorder columns for readability
        cols = ["pair", "ml_model", "balanced_accuracy", "analytic_p", "analytic_q", "perm_p", "perm_q", "perm_mean", "perm_std", "test_bacc",
                "cv_mean_models", "cv_max_models", "cv_best_model", "perm_p_best_model", "best_model_p", "best_model_q", "best_model_p_source",
                "row_mean_z", "row_mean_p", "row_mean_q", "row_mean_p_analytic", "row_mean_p_indep",
                "n_models_above_chance", "n_models_above_1sd", "ensemble_bacc", "ensemble_p", "ensemble_q",
                "subjects_correct_by_all_models", "n_subjects_cv", "n_permutations", "row_n_models",
                "test_mean_models", "test_max_models", "test_of_best_cv_model", "min_of_means",
                "test_sel_z", "test_sel_p", "test_sel_q", "test_row_mean_z", "test_row_mean_p", "test_row_mean_q",
                "n_train_0", "n_train_1", "n_test_0", "n_test_1",
                "distance", "rsa_rank", "rsa_p", "rsa_q", "rsa_consistent",
                "group_0_name", "group_1_name", "group_0_intra", "group_1_intra",
                "nm_prop_1", "group_0", "group_1"]
        df_results = df_results[[c for c in cols if c in df_results.columns]]

        sort_col = "min_of_means" if "min_of_means" in df_results.columns else "cv_mean_models"
        df_results = df_results.sort_values(by=[sort_col, "balanced_accuracy"], ascending=False)

        csv_path = os.path.join(self.output_directory, "all_pairs_summary.csv")
        os.makedirs(self.output_directory, exist_ok=True)
        df_results.to_csv(csv_path, index=False)
        print(f"Summary CSV saved to: {csv_path}")
        overview = self.write_pairs_overview(df_results)
        self.write_significance_summary(df_results, significance)

        print("Generating visualization...")
        self.generate_visualizations(df_results)
        if df_test is not None:
            self.generate_visualizations(df_results, value_col="test_bacc",
                                         title="Test-Set Balanced Accuracy by Pair and Model",
                                         filename="summary_test_balanced_accuracy.png")
        print(f"Visualization saved in the '{self.output_directory}' folder.")

        if self.test_directory:
            # ROI importance was already extracted by mode 'stat' from the saved
            # models; here only filter it to the pairs that passed the test.
            self.write_roi_importance_for_selected_pairs(overview)
            return

        print("Extracting ROI importance from saved ML models...")
        roi_df = summarize_roi_importance(
            base_directory=self.base_directory,
            output_directory=self.output_directory,
            pc_str=self.pc_num,
            extra_features_set=self.extra_features_set,
            n_windows_per_exp=self.n_windows_per_exp,
        )
        if roi_df is not None and not roi_df.empty:
            top_per_model = (
                roi_df[roi_df["rank"] <= 5]
                .groupby("ml_model")["roi_label"]
                .apply(lambda s: ", ".join(s.head(3).unique()))
            )
            print("Top ROIs by model (rank ≤ 5):")
            for model, labels in top_per_model.items():
                print(f"  {model}: {labels}")


def main(base_dir, align_dir, pc_str, extra_features_set=1, n_windows_per_exp=6, test_dir=None):
    stat = Statistics(
        base_dir,
        align_dir,
        pc_str,
        extra_features_set=extra_features_set,
        n_windows_per_exp=n_windows_per_exp,
        test_directory=test_dir,
    )
    stat.create_summary_files()

if __name__ == '__main__':
    base_dir = sys.argv[1]
    align_dir = sys.argv[2]
    pc_str = sys.argv[3]
    extra_set = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    n_windows = int(sys.argv[5]) if len(sys.argv) > 5 else 6
    main(base_dir, align_dir, pc_str, extra_set, n_windows)
