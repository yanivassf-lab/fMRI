"""
Interactive Dash dashboard for exploring ROI importance analysis results.

Enables selection of pairs, algorithms, and top-N ROIs with visualization
of common influential ROIs and comparative statistics.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Optional

import dash
from dash import dcc, html, Input, Output, State, dash_table
import plotly.graph_objects as go
import plotly.express as px


class ROIExplorerDashboard:
    """Interactive dashboard for ROI importance analysis."""

    def __init__(self, csv_path: str, port: int = 8050, debug: bool = False):
        """
        Initialize the dashboard.

        Parameters
        ----------
        csv_path : str
            Path to roi_importance_all.csv
        port : int
            Port to run the dashboard on
        debug : bool
            Enable debug mode
        """
        self.csv_path = csv_path
        self.port = port
        self.debug = debug

        # Load data with robust handling for pandas/numpy compatibility issues
        self.df = self._load_csv(csv_path)

        # ==========================================
        # ADDED: Global Importance Normalization
        # ==========================================
        # Normalize importance to [0, 1] per specific model run (pair + algorithm)
        # This completely removes magnitude bias when averaging across models or pairs
        self.df['importance'] = self.df.groupby(['pair', 'ml_model'])['importance'].transform(
            lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
        )
        # Extract unique pairs and algorithms
        self.pairs = sorted(self.df["pair"].unique())
        self.algorithms = sorted(self.df["ml_model"].unique())

        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self._setup_layout()
        self._setup_callbacks()

    @staticmethod
    def _load_csv(csv_path: str) -> pd.DataFrame:
        """
        Load CSV with robust handling for pandas/numpy compatibility issues.

        Parameters
        ----------
        csv_path : str
            Path to CSV file

        Returns
        -------
        pd.DataFrame
            Loaded data

        Raises
        ------
        FileNotFoundError
            If CSV file not found
        """
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        # Try standard loading first
        try:
            return pd.read_csv(csv_path)
        except Exception as e:
            pass

        # Fallback: use polars if available
        try:
            import polars as pl

            return pl.read_csv(csv_path).to_pandas()
        except ImportError:
            pass

        # Final fallback: manual loading with csv module
        import csv

        rows = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Convert numeric fields
                try:
                    row["roi_index"] = int(row["roi_index"])
                except (ValueError, KeyError):
                    pass
                try:
                    row["importance"] = float(row["importance"])
                except (ValueError, KeyError):
                    pass
                try:
                    row["rank"] = int(row["rank"])
                except (ValueError, KeyError):
                    pass
                try:
                    row["feature_set"] = int(row["feature_set"])
                except (ValueError, KeyError):
                    pass
                rows.append(row)

        return pd.DataFrame(rows)

    def _setup_layout(self):
        """Setup the dashboard layout."""
        self.app.layout = html.Div(
            [
                html.H1("ROI Importance Explorer", style={"textAlign": "center", "marginBottom": 30}),

                # Controls section
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Select Pairs:", style={"fontWeight": "bold"}),
                                dcc.Dropdown(
                                    id="pairs-dropdown",
                                    options=[{"label": p, "value": p} for p in self.pairs],
                                    value=self.pairs[:1] if self.pairs else [],
                                    multi=True,
                                    style={"width": "100%"},
                                ),
                            ],
                            style={"flex": 1, "marginRight": 20},
                        ),
                        html.Div(
                            [
                                html.Label("Select Algorithms:", style={"fontWeight": "bold"}),
                                dcc.Dropdown(
                                    id="algorithms-dropdown",
                                    options=[{"label": a, "value": a} for a in self.algorithms],
                                    value=self.algorithms,
                                    multi=True,
                                    style={"width": "100%"},
                                ),
                            ],
                            style={"flex": 1, "marginRight": 20},
                        ),
                        html.Div(
                            [
                                html.Label("Top N ROIs:", style={"fontWeight": "bold"}),
                                dcc.Slider(
                                    id="top-n-slider",
                                    min=1,
                                    max=30,
                                    step=1,
                                    value=10,
                                    marks={i: str(i) for i in range(1, 31, 3)},
                                    tooltip={"placement": "bottom", "always_visible": True},
                                ),
                            ],
                            style={"flex": 1},
                        ),
                    ],
                    style={
                        "display": "flex",
                        "gap": "20px",
                        "marginBottom": 30,
                        "padding": "20px",
                        "backgroundColor": "#f5f5f5",
                        "borderRadius": "8px",
                    },
                ),

                # Summary statistics
                html.Div(
                    [
                        html.Div(
                            id="summary-stats",
                            style={
                                "backgroundColor": "#e3f2fd",
                                "padding": "15px",
                                "borderRadius": "8px",
                                "marginBottom": 20,
                            },
                        ),
                    ]
                ),

                # Charts section
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Graph(id="top-rois-bar-chart", style={"height": "500px"})
                            ],
                            style={"flex": 1},
                        ),
                        html.Div(
                            [
                                dcc.Graph(id="roi-frequency-chart", style={"height": "500px"})
                            ],
                            style={"flex": 1},
                        ),
                    ],
                    style={"display": "flex", "gap": "20px", "marginBottom": 20},
                ),

                # Comparison charts
                html.Div(
                    [
                        html.Div(
                            [
                                dcc.Graph(id="algorithm-comparison", style={"height": "500px"})
                            ],
                            style={"flex": 1},
                        ),
                        html.Div(
                            [
                                dcc.Graph(id="pair-comparison", style={"height": "500px"})
                            ],
                            style={"flex": 1},
                        ),
                    ],
                    style={"display": "flex", "gap": "20px", "marginBottom": 20},
                ),

                # ==========================================
                # ADDED SECTION: Research Insights & Cross-Pair Summary
                # ==========================================
                html.Div(
                    [
                        html.H3("Research Insights: Cross-Pair Consensus & ROI Summary", style={"marginBottom": 15}),
                        html.P(
                            "This section summarizes the overall model consensus across all pairs and highlights globally influential brain regions.",
                            style={"color": "#555", "marginBottom": 20},
                        ),
                        html.Div(
                            [
                                html.Div(
                                    [
                                        dcc.Graph(id="global-pair-consensus-chart",
                                                  style={"height": "100%", "minHeight": "500px"})
                                    ],
                                    style={"flex": 1, "minWidth": "300px"},
                                ),
                                html.Div(
                                    [
                                        dcc.Graph(id="global-roi-summary-chart",
                                                  style={"height": "100%", "minHeight": "500px"})
                                    ],
                                    style={"flex": 1, "minWidth": "300px"},
                                ),
                            ],
                            style={"display": "flex", "flexWrap": "wrap", "gap": "20px"},
                            # flexWrap מונע פלישה ודחיקה למטה במסכים צרים
                        ),
                    ],
                    style={
                        "padding": "20px",
                        "backgroundColor": "#fff3e0",
                        "borderRadius": "8px",
                        "marginBottom": 20,
                        "border": "1px solid #ffe0b2",
                        "boxSizing": "border-box",
                    },
                ),

                # Detailed table
                html.Div(
                    [
                        html.H3("Detailed ROI Rankings (Full Data)"),
                        html.H4("Please note that while the raw importance scores in the original CSV file are not normalized, all data displayed in the dashboard graphs is normalized.\n"
                                "This allows for direct and fair comparisons across different machine learning models."),
                        html.P(
                            "This table contains all ROIs for the selected pairs and algorithms, ignoring the Top-N filter. "
                            "Use the text boxes below the headers to filter (e.g., '> 0.5' or 'Drums'), and click headers to sort.",
                            style={"fontSize": "14px", "color": "#666"}
                        ),
                        # Use dash_table.DataTable for native sorting, filtering, and efficient pagination
                        dash_table.DataTable(
                            id="detailed-table",
                            columns=[
                                {"name": "Pair", "id": "pair"},
                                {"name": "Algorithm", "id": "ml_model"},
                                {"name": "ROI Label", "id": "roi_label"},
                                {"name": "ROI Index", "id": "roi_index"},
                                {"name": "Importance", "id": "importance"},
                                {"name": "Rank", "id": "rank"},
                            ],
                            filter_action="native",  # Enables per-column filtering
                            sort_action="native",  # Enables per-column sorting
                            page_action="native",  # Enables pagination to handle large datasets efficiently
                            page_size=20,  # Display 20 rows per page
                            style_table={"overflowX": "auto"},
                            style_cell={
                                "textAlign": "left",
                                "padding": "10px",
                                "fontFamily": "Arial, sans-serif"
                            },
                            style_header={
                                "backgroundColor": "#f2f2f2",
                                "fontWeight": "bold",
                                "border": "1px solid #ddd"
                            },
                            style_data_conditional=[
                                {
                                    "if": {"row_index": "odd"},
                                    "backgroundColor": "#f9f9f9"
                                }
                            ]
                        ),
                    ],
                    style={"padding": "20px", "backgroundColor": "#f5f5f5", "borderRadius": "8px"},
                ),
            ],
            style={"padding": "20px", "fontFamily": "Arial, sans-serif", "maxWidth": "1600px", "margin": "0 auto"},
        )

    def _setup_callbacks(self):
        """Setup interactive callbacks."""

        @self.app.callback(
            [
                Output("summary-stats", "children"),
                Output("top-rois-bar-chart", "figure"),
                Output("roi-frequency-chart", "figure"),
                Output("algorithm-comparison", "figure"),
                Output("pair-comparison", "figure"),
                Output("global-pair-consensus-chart", "figure"),
                Output("global-roi-summary-chart", "figure"),
                Output("detailed-table", "data"),
            ],
            [
                Input("pairs-dropdown", "value"),
                Input("algorithms-dropdown", "value"),
                Input("top-n-slider", "value"),
            ],
        )
        def update_dashboard(selected_pairs, selected_algorithms, top_n):
            """Update all dashboard elements based on selections."""
            if not selected_pairs or not selected_algorithms:
                return (
                    html.Div("Please select at least one pair and one algorithm."),
                    go.Figure(),
                    go.Figure(),
                    go.Figure(),
                    go.Figure(),
                    go.Figure(),
                    go.Figure(),
                    html.Div("No data to display."),
                )

            # Ensure selections are lists
            if isinstance(selected_pairs, str):
                selected_pairs = [selected_pairs]
            if isinstance(selected_algorithms, str):
                selected_algorithms = [selected_algorithms]

            # Filter data
            filtered_df = self.df[
                (self.df["pair"].isin(selected_pairs)) & (self.df["ml_model"].isin(selected_algorithms))
            ].copy()

            if filtered_df.empty:
                return []

            # Get top N ROIs
            top_rois_df = filtered_df[filtered_df["rank"] <= top_n].copy()

            # Summary statistics
            summary_text = self._generate_summary_stats(filtered_df, top_rois_df, selected_pairs, selected_algorithms, top_n)

            # Top ROIs bar chart
            top_rois_bar = self._create_top_rois_chart(top_rois_df)

            # ROI frequency chart
            roi_freq_chart = self._create_roi_frequency_chart(top_rois_df)

            # Algorithm comparison
            algo_comparison = self._create_algorithm_comparison(top_rois_df)

            # Pair comparison
            pair_comparison = self._create_pair_comparison(top_rois_df)

            # ADDED: Global Research Summary Charts (Good/Bad pairs consensus & overall ROI summary)
            global_pair_consensus = self._create_global_pair_consensus_chart(top_n)
            global_roi_summary = self._create_global_roi_summary_chart(top_n)

            # Use filtered_df (which contains ALL ranks for the selected pairs and algorithms),
            # specifically ignoring top_rois_df so the table shows the entire dataset.
            display_df = filtered_df[["pair", "ml_model", "roi_label", "roi_index", "importance", "rank"]].sort_values(
                ["pair", "ml_model", "rank"])

            # Format the importance column for cleaner display in the table
            display_df["importance"] = display_df["importance"].round(6)

            # Convert the dataframe to a list of dictionaries required by dash_table
            table_data = display_df.to_dict("records")

            return (
                summary_text,
                top_rois_bar,
                roi_freq_chart,
                algo_comparison,
                pair_comparison,
                global_pair_consensus,
                global_roi_summary,
                # Pass the raw data dictionary to the interactive table
                table_data,
            )

    def _generate_summary_stats(self, filtered_df, top_rois_df, selected_pairs, selected_algorithms, top_n):
        """Generate summary statistics."""
        total_analyses = len(filtered_df.groupby(["pair", "ml_model"]))
        unique_rois = len(top_rois_df["roi_index"].unique())

        # Most common ROI in top-N
        roi_counts = top_rois_df.groupby(["roi_label", "roi_index"]).size().reset_index(name="count").sort_values("count", ascending=False)
        top_roi = roi_counts.iloc[0] if not roi_counts.empty else None

        stats = [
            html.H4("Summary Statistics"),
            html.Div(
                [
                    html.Span(f"Pairs: {len(selected_pairs)}", style={"marginRight": 20}),
                    html.Span(f"Algorithms: {len(selected_algorithms)}", style={"marginRight": 20}),
                    html.Span(f"Total Analyses: {total_analyses}", style={"marginRight": 20}),
                    html.Span(f"Unique ROIs in Top-{top_n}: {unique_rois}", style={"marginRight": 20}),
                    html.Span(
                        f"Most Common ROI: {top_roi['roi_label']} (appears {top_roi['count']} times)"
                        if top_roi is not None
                        else "N/A",
                        style={"fontWeight": "bold"},
                    ),
                ]
            ),
        ]
        return stats

    def _create_top_rois_chart(self, top_rois_df):
        """Create bar chart of top ROIs."""
        roi_importance = (
            top_rois_df.groupby(["roi_label", "roi_index"])["importance"]
            .mean()
            .reset_index()
            .sort_values("importance", ascending=False)
        )

        fig = go.Figure(
            data=[
                go.Bar(
                    x=roi_importance["roi_label"][:20],
                    y=roi_importance["importance"][:20],
                    text=roi_importance["roi_index"][:20],
                    textposition="outside",
                    marker=dict(color=roi_importance["importance"][:20], colorscale="Viridis"),
                    hovertemplate="<b>%{x}</b><br>Avg Importance: %{y:.4f}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title="Top 20 ROIs by Average Importance",
            xaxis_title="ROI Label",
            yaxis_title="Average Importance",
            height=500,
            xaxis_tickangle=-45,
            showlegend=False,
        )
        return fig

    def _create_roi_frequency_chart(self, top_rois_df):
        """Create chart of ROI frequency in top-N across analyses."""
        roi_freq = (
            top_rois_df.groupby(["roi_label", "roi_index"])
            .size()
            .reset_index(name="frequency")
            .sort_values("frequency", ascending=False)
            .head(20)
        )

        fig = go.Figure(
            data=[
                go.Bar(
                    x=roi_freq["roi_label"],
                    y=roi_freq["frequency"],
                    text=roi_freq["roi_index"],
                    textposition="outside",
                    marker=dict(color=roi_freq["frequency"], colorscale="Blues"),
                    hovertemplate="<b>%{x}</b><br>Appearances: %{y}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title="ROI Frequency in Top-N (all pair/algorithm combinations)",
            xaxis_title="ROI Label",
            yaxis_title="Frequency",
            height=500,
            xaxis_tickangle=-45,
            showlegend=False,
        )
        return fig

    def _create_algorithm_comparison(self, top_rois_df):
        """Create algorithm comparison chart."""
        algo_summary = (
            top_rois_df.groupby("ml_model")["importance"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )

        fig = go.Figure(
            data=[
                go.Bar(
                    x=algo_summary["ml_model"],
                    y=algo_summary["mean"],
                    error_y=dict(type="data", array=algo_summary["std"]),
                    text=algo_summary["count"],
                    texttemplate="n=%{text}",
                    textposition="outside",
                    marker=dict(color=algo_summary["mean"], colorscale="Plasma"),
                    hovertemplate="<b>%{x}</b><br>Avg Importance: %{y:.4f}<br>Std: %{error_y.array:.4f}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title="Algorithm Comparison (Mean Importance)",
            xaxis_title="Algorithm",
            yaxis_title="Mean Importance",
            height=500,
            showlegend=False,
        )
        return fig

    def _create_pair_comparison(self, top_rois_df):
        """Create pair comparison chart."""
        pair_summary = (
            top_rois_df.groupby("pair")["importance"]
            .agg(["mean", "std", "count"])
            .reset_index()
            .sort_values("mean", ascending=False)
        )

        fig = go.Figure(
            data=[
                go.Bar(
                    x=pair_summary["pair"],
                    y=pair_summary["mean"],
                    error_y=dict(type="data", array=pair_summary["std"]),
                    text=pair_summary["count"],
                    texttemplate="n=%{text}",
                    textposition="outside",
                    marker=dict(color=pair_summary["mean"], colorscale="Magma"),
                    hovertemplate="<b>%{x}</b><br>Avg Importance: %{y:.4f}<br>Std: %{error_y.array:.4f}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title="Pair Comparison (Mean Importance)",
            xaxis_title="Pair",
            yaxis_title="Mean Importance",
            height=500,
            xaxis_tickangle=-45,
            showlegend=False,
        )
        return fig

    # ==========================================
    # ADDED HELPER METHODS FOR RESEARCH INSIGHTS
    # ==========================================
    def _create_global_pair_consensus_chart(self, top_n):
        """Create a chart summarizing the consensus (good vs bad/noisy pairs) across all pairs."""
        top_df = self.df[self.df["rank"] <= top_n]
        consensus_df = top_df.groupby(["pair", "roi_label"]).size().reset_index(name="freq")
        high_consensus = consensus_df[consensus_df["freq"] >= 4].groupby("pair").size().reset_index(
            name="high_consensus_count")

        # שימוש בכל הזוגות במערכת מבלי לפספס אף אחד
        all_pairs_df = pd.DataFrame({"pair": self.pairs})
        summary_df = pd.merge(all_pairs_df, high_consensus, on="pair", how="left").fillna(0)
        # מיון מהגבוה לנמוך כדי שהזוגות המובילים (כמו NM vs mus) יופיעו למעלה
        summary_df = summary_df.sort_values(by="high_consensus_count", ascending=True)

        fig = go.Figure(
            data=[
                go.Bar(
                    y=summary_df["pair"],
                    x=summary_df["high_consensus_count"],
                    orientation="h",
                    marker=dict(
                        color=summary_df["high_consensus_count"],
                        colorscale="Tealgrn",
                    ),
                    hovertemplate="<b>%{y}</b><br>High Consensus ROIs (>=4 models): %{x}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title="Pair Consensus Ranking (Robust Signals vs. Noise)",
            xaxis_title="Number of High-Consensus ROIs (Agreement across >=4 models)",
            yaxis_title="Instrument Pair",
            height=750,  # גובה מותאם אישית המכיל את כל 22 הזוגות ללא גלישה או חיתוך
            margin=dict(l=150),  # מרווח שמאלי שמונע חיתוך של שמות הזוגות הארוכים
            yaxis=dict(tickfont=dict(size=11)),
        )
        return fig

    def _create_global_roi_summary_chart(self, top_n):
        """Create a global summary chart of top influential ROIs across all pairs and models."""
        top_df = self.df[self.df["rank"] <= top_n]
        roi_summary = (
            top_df.groupby(["roi_label", "roi_index"])
            .size()
            .reset_index(name="global_frequency")
            .sort_values("global_frequency", ascending=False)
            .head(15)
        )

        fig = go.Figure(
            data=[
                go.Bar(
                    x=roi_summary["roi_label"],
                    y=roi_summary["global_frequency"],
                    text=roi_summary["roi_index"],
                    textposition="outside",
                    marker=dict(color=roi_summary["global_frequency"], colorscale="Sunset"),
                    hovertemplate="<b>%{x}</b><br>Global Frequency: %{y}<extra></extra>",
                )
            ]
        )
        fig.update_layout(
            title="Top Global Influential ROIs (Across All Analyses)",
            xaxis_title="ROI Label",
            yaxis_title="Total Frequency in Top-N",
            height=450,
            xaxis_tickangle=-45,
        )
        return fig


    def run(self):
        """Run the dashboard server."""
        print(f"\n{'='*60}")
        print("ROI Importance Explorer Dashboard")
        print(f"{'='*60}")
        print(f"Starting dashboard at http://localhost:{self.port}")
        print(f"Data loaded from: {self.csv_path}")
        print(f"Total records: {len(self.df)}")
        print(f"Pairs: {len(self.pairs)}")
        print(f"Algorithms: {len(self.algorithms)}")
        print(f"\nPress Ctrl+C to stop the server")
        print(f"{'='*60}\n")

        self.app.run(debug=self.debug, port=self.port, host="0.0.0.0")


def launch_roi_explorer(csv_path: str, port: int = 8050, debug: bool = False):
    """
    Launch the ROI Importance Explorer dashboard.

    Parameters
    ----------
    csv_path : str
        Path to roi_importance_all.csv file
    port : int, optional
        Port to run the dashboard on (default: 8050)
    debug : bool, optional
        Enable debug mode (default: False)

    Examples
    --------
    >>> launch_roi_explorer('/path/to/roi_importance_all.csv')
    """
    dashboard = ROIExplorerDashboard(csv_path, port=port, debug=debug)
    dashboard.run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Launch ROI Importance Explorer Dashboard")
    parser.add_argument(
        "csv_path",
        help="Path to roi_importance_all.csv",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Port to run the dashboard on (default: 8050)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )

    args = parser.parse_args()

    if not os.path.exists(args.csv_path):
        print(f"Error: File not found: {args.csv_path}")
        exit(1)

    launch_roi_explorer(args.csv_path, port=args.port, debug=args.debug)
