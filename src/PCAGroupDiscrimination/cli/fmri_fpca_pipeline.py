#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import json
from itertools import combinations

from PCAGroupDiscrimination.stat import Statistics
from PCAGroupDiscrimination.build_signals_from_files import setup_logger
from PCAGroupDiscrimination.process_signals import ProcessSignals
from PCAGroupDiscrimination.signal_aligner import SignalAligner
from PCAGroupDiscrimination.ml_analyzer import MLAnalyzer
from PCAGroupDiscrimination.ml_tester import MLTester

# =========================================================
# App
# =========================================================

class PipelineApp:
    def __init__(self, args):
        self.args = args
        os.makedirs(args.output_dir, exist_ok=True)

    def run_align_signals(self, X, ids, atlas_labels):
        return SignalAligner(self.args).align(X, ids, atlas_labels)

    def run_ml(self, X, ids, atlas_labels):
        ml_analyzer = MLAnalyzer(self.args)
        for model in self.args.ml_models:
            for zero_class, one_class in combinations(self.args.groups, 2):
                if (zero_class != "NM" and one_class == "mus") or (zero_class == "mus" and one_class != "NM"):
                    continue
                dict_out = ml_analyzer.analyze(X, ids, atlas_labels, model, zero_class, one_class)
        return

    def run_ml_test(self, X, ids):

        # Verify that the user provided a valid JSON file for the test pairs
        if not self.args.test_pairs_file or not os.path.exists(self.args.test_pairs_file):
            raise FileNotFoundError(
                "Test mode requires a valid JSON file path provided via --test-pairs-file."
            )

        # Load the specific pairs to test from the JSON file
        with open(self.args.test_pairs_file, 'r') as f:
            test_config = json.load(f)

        pairs_to_test = test_config.get("test_pairs", [])

        if not pairs_to_test:
            self.logger.warning("No test pairs found in the JSON file. Exiting test mode.")
            return

        ml_tester = MLTester(self.args)

        for model in self.args.ml_models:
            for pair in pairs_to_test:
                # Ensure the pair is exactly a couple of classes
                if len(pair) != 2:
                    self.logger.warning(f"Invalid pair format in JSON: {pair}. Skipping.")
                    continue

                zero_class, one_class = pair[0], pair[1]

                # Execute the test strictly for the specified pair
                ml_tester.test(X, ids, model, zero_class, one_class)
        return

    def create_stat(self):
        base_dir = os.path.join(self.args.output_dir, f'2_ml_{self.args.pc_str}')
        align_dir = os.path.join(self.args.output_dir, f'1_align_signals')
        stat = Statistics(base_dir, align_dir, self.args.pc_str)
        stat.create_summary_files()

    def run(self):
        self.logger = setup_logger(self.args.output_dir)
        param_exists = False
        if self.args.mode in ["align-signals", "ml", "ml-test"]:
            X, ids, atlas_labels = ProcessSignals(self.args, self.logger).read_signals()
        if self.args.mode == "align-signals":
            self.run_align_signals(X, ids, atlas_labels)
            param_exists = True
        if self.args.mode == "ml":
            self.run_ml(X, ids, atlas_labels)
            param_exists = True
        if self.args.mode == "stat":
            self.create_stat()
            param_exists = True
        if self.args.mode == "ml-test":
            self.run_ml_test(X, ids)
            param_exists = True

        if not param_exists:
            raise ValueError(f"Unknown mode: {self.args.mode}")


# =========================================================
# Argparse
# =========================================================

def build_parser():
    p = argparse.ArgumentParser(description="Unified clustering / stability / features / ML pipeline")

    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--metadata-csv", required=True)
    p.add_argument("--ml-hyperparameters-file", required=False, type=str,
                   help="File with ML parameters")
    p.add_argument("--test-pairs-file", type=str, default=None,
                   help="Path to JSON file containing specific pairs to evaluate in test mode.")
    p.add_argument("--mode", required=True, type=str,
                   choices=["align-signals", "ml", "stat", "ml-test"])
    p.add_argument("--file-name-pattern", default=r"(.+)_ses-[A-Za-z0-9]+.*?movement(\d+)")
    p.add_argument("--n-pcs", type=int, default=7)
    p.add_argument("--target-pc-index", type=int, default=0)
    p.add_argument("--row-normalize-ml", action="store_true", help="Normalize the examples")
    p.add_argument("--n-permutations", type=int, default=200,
                   help="Number of label permutations for ML permutation test")
    p.add_argument("--extra-features-set", type=int, default=1, choices=[0, 1, 2],
                   help="Temporal feature extraction: 0=None, 1=Fixed 3-window split, "
                        "2=Event-driven music transitions.")
    p.add_argument("--mov1-transition-trs", nargs='+', default=[40, 201, 433, 481, 570],
                   help="Transition times in movement 1")
    p.add_argument("--mov2-transition-trs", nargs='+', default=[114, 304, 313, 450],
                   help="Transition times in movement 2")
    p.add_argument("--ml-models", nargs="+", default=["LR", "SVM", "NN", "DTree", "RandForest"],
                   help="Choice of ML model. 'NN' uses an internal PyTorch network wrapper")
    p.add_argument("--groups", nargs="+",
                   default=["Guitars", "Vocal", "Wind", "Drums", "Strings", "Keyboards", "NM", "mus"],
                   help="List of groups to analyze. Use 'mus' to include all musicians as one group vs Nm.")
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument(
        "--cache-file-pardir",
        type=str,
        default=None,
        help="Directory for the cache file. After the first run, the input signals are stored here in "
             "a cache file. Default: input-dir"
    )

    # Parameters for working on raw data instead of recovered signals from PCs.
    p.add_argument(
        "--use-raw-data",
        action="store_true",
        help="Use the raw data instead of the recovered signals from the PCs."
    )
    p.add_argument(
        "--raw-data-path",
        type=str,
        default=None,
        help="If the cache file does not exist, read the raw data from this directory"
    )
    p.add_argument("--TR", type=float, default=0.75, help="Repetition time (TR).")
    p.add_argument("--smooth-size", type=int, default=5, help="Box size of smoothing kernel.")
    p.add_argument("--highpass", type=float, default=0.01, help="High-pass filter cutoff.")
    p.add_argument("--lowpass", type=float, default=0.08, help="Low-pass filter cutoff.")
    p.add_argument("--use-nilearn-filter", action='store_true', help="Use Nilearn for filtering.")
    p.add_argument("--n-compcor-nilearn-filter", type=int, default=5, help="Number of compcor components for Nilearn filtering.")
    p.add_argument("--smoothing-fwhm-nilearn-filter", type=float, default=6.0, help="FWHM for smoothing in Nilearn filtering.")
    p.add_argument("--processed", action='store_true', help="Data is already preprocessed.")
    p.add_argument("--n-skip-vols-start", type=int, default=0, help="Vols to discard from start.")
    p.add_argument("--n-skip-vols-end", type=int, default=0, help="Vols to discard from end.")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "ml-test" and not args.test_pairs_file:
        parser.error("--test-pairs-file is required when mode is 'ml-test'.")
    elif args.mode == "ml" and not args.ml_hyperparameters_file:
        parser.error("--ml-hyperparameters-file is required when mode is 'ml'.")
    if args.mode == "ml" and not os.path.exists(args.ml_hyperparameters_file):
        parser.error(f"ML hyperparameters file '{args.ml_hyperparameters_file}' does not exist.")
    if args.mode == "ml-test" and not os.path.exists(args.test_pairs_file):
        parser.error(f"Test pairs file '{args.test_pairs_file}' does not exist.")
    if args.use_raw_data and not args.raw_data_path:
        parser.error("--raw-data-path is required when --use-raw-data is set.")
    args.target_pc_index = 'raw' if args.use_raw_data else args.target_pc_index
    args.pc_str = 'raw' if args.use_raw_data else f"pc-{args.target_pc_index}"
    args.cache_file_pardir = args.input_dir if not args.cache_file_pardir else args.cache_file_pardir
    app = PipelineApp(args)
    app.run()


if __name__ == "__main__":
    main()
