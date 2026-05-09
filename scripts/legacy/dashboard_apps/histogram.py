"""Standalone Dash app for interactively zooming into p-value histograms."""

import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import numpy as np

# Load the CSV data into a DataFrame
data = pd.read_csv("granger_causality_results_benito_Human.csv")

# Extract the p-values column
p_values = data['p-value']

# Define bin edges that strictly cover the range [0, 1]
bin_edges = np.linspace(0, 1, 21)  # Creates 20 bins between 0 and 1

# Initialize the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)
server = app.server

# Define the app layout
app.layout = html.Div([
    html.H1("Interactive P-value Distribution Analysis", style={'font-size': '60px'}),
    html.P("Click on a bar in the histogram to zoom into that range. Use the button to enlarge the histogram in a new tab.",
           style={'font-size': '32px'}),
    html.Div([
        html.Div([
            dcc.Graph(id='histogram', style={'height': '60vh', 'width': '80vw'})
        ]),
        html.Div([
            dcc.Graph(id='expanded_histogram', style={'height': '60vh', 'width': '80vw'})
        ]),
        html.Div([
            dcc.Graph(id='further_expanded_histogram', style={'height': '60vh', 'width': '80vw'})
        ]),
        html.Div([
            dcc.Graph(id='furthest_expanded_histogram', style={'height': '60vh', 'width': '80vw'})
        ]),
    ], style={'display': 'flex', 'flex-direction': 'row', 'justify-content': 'space-around'}),
])


# Main Histogram Callback
@app.callback(
    Output('histogram', 'figure'),
    Input('expanded_histogram', 'id')
)
def display_histogram(_):
    """Render the top-level p-value histogram."""
    hist_original = go.Histogram(
        x=p_values,
        xbins=dict(start=0, end=1, size=(bin_edges[1] - bin_edges[0])),
        marker_color='blue'
    )

    fig = go.Figure(data=[hist_original])
    fig.update_layout(
        title="P-value Distribution",
        xaxis_title="P-value",
        yaxis_title="Count",
        bargap=0.1,
    )

    return fig


# General Histogram Expansion Callback
def generate_expanded_histogram(click_data, bin_edges, p_values, zoom_level=1):
    """Generate an expanded histogram for any zoom level."""
    if click_data is None:
        return go.Figure()

    base_bin_size = bin_edges[1] - bin_edges[0]
    bin_size = base_bin_size / (20 ** (zoom_level - 1))

    x_center = click_data['points'][0]['x']
    bin_start = np.floor(x_center / bin_size) * bin_size
    bin_end = bin_start + bin_size

    filtered_p_values = p_values[(p_values >= bin_start) & (p_values < bin_end)]

    histogram = go.Histogram(
        x=filtered_p_values,
        xbins=dict(start=bin_start, end=bin_end, size=bin_size / 20),
        marker_color={1: 'blue', 2: 'orange', 3: 'green'}.get(zoom_level, 'gray')
    )

    fig = go.Figure(data=[histogram])
    fig.update_layout(
        title=f"Expanded View (Level {zoom_level}): {bin_start:.6f} to {bin_end:.6f}",
        xaxis_title="P-value",
        yaxis_title="Count",
        xaxis=dict(
            range=[bin_start, bin_end],
            tickformat=f".{6*zoom_level}f"
        ),
        bargap=0.1,
    )

    return fig


@app.callback(
    Output('expanded_histogram', 'figure'),
    Input('histogram', 'clickData')
)
def update_expanded_histogram(click_data):
    """Render the first zoom level from the main histogram selection."""
    return generate_expanded_histogram(click_data, bin_edges, p_values, zoom_level=1)


@app.callback(
    Output('further_expanded_histogram', 'figure'),
    Input('expanded_histogram', 'clickData')
)
def update_further_expanded_histogram(click_data):
    """Render the second zoom level from the prior histogram selection."""
    return generate_expanded_histogram(click_data, bin_edges, p_values, zoom_level=2)

@app.callback(
    Output('furthest_expanded_histogram', 'figure'),
    Input('further_expanded_histogram', 'clickData')
)
def update_furthest_expanded_histogram(click_data):
    """Render the third zoom level from the prior histogram selection."""
    return generate_expanded_histogram(click_data, bin_edges, p_values, zoom_level=3)

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
