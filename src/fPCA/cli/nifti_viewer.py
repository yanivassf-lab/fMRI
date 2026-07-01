#!/usr/bin/env python

import argparse

import dash
import nibabel as nib
import numpy as np
import plotly.express as px
from dash import dcc, html, Input, Output

def main():
    # Add argparse to handle input arguments
    parser = argparse.ArgumentParser(description="Interactive NIfTI Slice Viewer")
    parser.add_argument(
        "--nifti-file",
        type=str,
        required=True,
        help="Path to the NIfTI file to visualize."
    )
    args = parser.parse_args()

    # Load the NIfTI file
    nifti_file = args.nifti_file
    img = nib.load(nifti_file)
    data = img.get_fdata()

    # Check dimensions of the data
    is_4d = len(data.shape) == 4
    if is_4d:
        time_points = data.shape[3]  # Number of volumes in 4D data
    else:
        time_points = 1  # Treat as a single volume

    # Initialize the Dash app
    app = dash.Dash(__name__)

    # App layout
    app.layout = html.Div([
        html.H1("Interactive NIfTI Slice Viewer", style={'textAlign': 'center'}),
        dcc.Dropdown(
            id="axis-dropdown",
            options=[
                {"label": "Axial (Z-Axis)", "value": 2},
                {"label": "Coronal (Y-Axis)", "value": 1},
                {"label": "Sagittal (X-Axis)", "value": 0},
            ],
            value=2,  # Default to axial
            clearable=False,
            style={'width': '50%', 'margin': '0 auto', 'fontSize': '30px'}
        ),
        dcc.Dropdown(
            id="color-dropdown",
            options=[
                {"label": "Gray", "value": "gray"},
                {"label": "Viridis", "value": "viridis"},
                {"label": "Plasma", "value": "plasma"},
                {"label": "Inferno", "value": "inferno"},
                {"label": "Cividis", "value": "cividis"},
                {"label": "Magma", "value": "magma"},
            ],
            value="viridis",  # Default color scale
            clearable=False,
            style={'width': '50%', 'margin': '0 auto', 'fontSize': '30px'}
        ),

        html.Div([
            html.Label("Timepoint", className='slider-label'),
            dcc.Slider(
                id="time-slider",
                min=0,
                max=time_points - 1,
                step=1,
                value=0,
                marks={i: f"Vol {i}" for i in range(time_points)} if is_4d else {},
                tooltip={"placement": "bottom", "always_visible": True},
                included=False,
                className='slider'
            ),
        ], className='slider-container'),

        html.Div([
            html.Label("Slice", className='slider-label'),
            dcc.Slider(
                id="slice-slider",
                min=0,
                max=data.shape[2] - 1,  # Default for axial slices
                step=1,
                value=data.shape[2] // 2,  # Default to the middle slice
                marks={i: str(i) for i in range(0, data.shape[2], max(1, data.shape[2] // 10))},
                tooltip={"placement": "bottom", "always_visible": True},
                included=False,
                className='slider'
            ),
        ], className='slider-container'),

        dcc.Graph(id="slice-plot",
                  style={"height": "70vh"},  # Set graph height to 70% of the viewport height
                  ),
    ])


    # Update slider range and marks dynamically based on axis selection
    @app.callback(
        Output("slice-slider", "max"),
        Output("slice-slider", "marks"),
        Input("axis-dropdown", "value")
    )
    def update_slider(axis):
        max_slices = data.shape[axis] - 1
        marks = {i: str(i) for i in range(0, max_slices + 1, max(1, max_slices // 10))}
        return max_slices, marks


    # Update the slice visualization
    @app.callback(
        Output("slice-plot", "figure"),
        Input("axis-dropdown", "value"),
        Input("slice-slider", "value"),
        Input("time-slider", "value"),
        Input("color-dropdown", "value")
    )
    def update_slice(axis, slice_idx, time_idx, color_scale):
        # Handle 4D or 3D data
        if is_4d:
            volume_data = data[:, :, :, time_idx]
        else:
            volume_data = data

        # Extract the appropriate slice
        if axis == 0:  # Sagittal
            slice_data = volume_data[slice_idx, :, :]
        elif axis == 1:  # Coronal
            slice_data = volume_data[:, slice_idx, :]
        elif axis == 2:  # Axial
            slice_data = volume_data[:, :, slice_idx]
        else:
            raise ValueError("Invalid axis selected.")

        # Ensure the slice is 2D
        slice_data = np.squeeze(slice_data)  # Removes any singleton dimensions

        # Transpose and flip for correct orientation
        slice_data = np.flipud(slice_data.T)
        # slice_data = volume_data[:, :, slice_idx]

        # Create the figure
        fig = px.imshow(
            slice_data,
            # origin='lower',  # ← row 0 is at the bottom
            color_continuous_scale=color_scale,
            title=f"Axis {axis} | Slice {slice_idx} | Volume {time_idx}" if is_4d else f"Axis {axis} | Slice {slice_idx}",
            labels={"color": "Intensity"}
        )
        fig.update_layout(
            xaxis_title="X",
            yaxis_title="Y",
            template="plotly_dark"
        )
        return fig

    app.run(debug=True)

# Run the app
if __name__ == "__main__":
    main()

##############################################################################
# Examples:

# python nifti_viewer.py --nifti-file /path/to/your/file.nii.gz

# Python output:
# python nifti_viewer.py --nifti-file "/Users/user/Library/CloudStorage/GoogleDrive-refaelkohen@mail.tau.ac.il/My Drive/TLV-U-drive/BrainWork/results-share/full-output-nonlin-r-lin-r/AxSI/pasi.nii.gz"

# MATLAB output:
# python nifti_viewer.py --nifti-file "/Users/user/Library/CloudStorage/GoogleDrive-refaelkohen@mail.tau.ac.il/My Drive/TLV-U-drive/BrainWork/results-share/output-full-par/AxSI/ADD.nii.gz"
