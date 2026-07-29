"""
Command-line interface for launching the ROI Importance Explorer dashboard.

This script provides easy access to the interactive visualization tool
for analyzing ROI importance results from the fMRI pipeline.
"""

import os
import sys
import argparse
from pathlib import Path


def find_roi_importance_csv(search_path: str) -> str:
    """
    Search for roi_importance_all.csv in the directory and its parents.

    Parameters
    ----------
    search_path : str
        Starting directory to search from

    Returns
    -------
    str
        Path to roi_importance_all.csv if found

    Raises
    ------
    FileNotFoundError
        If roi_importance_all.csv is not found
    """
    search_dir = Path(search_path).resolve()

    # Search in current and parent directories
    for directory in [search_dir] + list(search_dir.parents):
        csv_file = directory / "roi_importance_all.csv"
        if csv_file.exists():
            return str(csv_file)

        # Also check for global_summary subdirectory
        summary_csv = directory / "global_summary" / "roi_importance_all.csv"
        if summary_csv.exists():
            return str(summary_csv)

    raise FileNotFoundError(
        f"Could not find roi_importance_all.csv in {search_dir} or its parent directories"
    )


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Launch interactive ROI Importance Explorer dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch with explicit CSV path
  python -m PCAGroupDiscrimination.cli.explore_roi /path/to/roi_importance_all.csv

  # Launch from ML directory (auto-find CSV)
  python -m PCAGroupDiscrimination.cli.explore_roi /path/to/2_ml_pc-1

  # Launch on custom port
  python -m PCAGroupDiscrimination.cli.explore_roi /path/to/csv --port 8080

  # Launch in debug mode
  python -m PCAGroupDiscrimination.cli.explore_roi /path/to/csv --debug
        """,
    )

    parser.add_argument(
        "path",
        help="Path to roi_importance_all.csv or parent directory to search in",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port to run the dashboard on (default: 8050)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode for development",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )

    args = parser.parse_args()

    # Determine CSV path
    if os.path.isfile(args.path):
        csv_path = args.path
    else:
        try:
            csv_path = find_roi_importance_csv(args.path)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Verify file exists
    if not os.path.exists(csv_path):
        print(f"Error: File not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # Import and launch dashboard
    try:
        from PCAGroupDiscrimination.roi_explorer_dashboard import ROIExplorerDashboard
    except ImportError as e:
        print(f"Error importing dashboard: {e}", file=sys.stderr)
        print("Make sure dash and plotly are installed: pip install dash plotly", file=sys.stderr)
        sys.exit(1)

    print(f"\nLaunching ROI Importance Explorer")
    print(f"{'='*60}")
    print(f"CSV file: {csv_path}")
    print(f"Host: {args.host}:{args.port}")
    print(f"Debug mode: {args.debug}")
    print(f"{'='*60}\n")

    dashboard = ROIExplorerDashboard(csv_path, port=args.port, debug=args.debug)
    dashboard.run()


if __name__ == "__main__":
    main()
