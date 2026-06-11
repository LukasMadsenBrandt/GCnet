"""Standalone Dash app for plotting selected gene expression time series."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import numpy as np
from gene_analysis.datasets.kutsche import load_and_preprocess_data as load_and_preprocess_kutsche
from gene_analysis.datasets.kutsche import preprocess_pipeline as preprocess_pipeline
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_daq as daq
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
import os

# Load and filter data
df = load_and_preprocess_kutsche(os.path.join('Data', 'Kutsche', 'genes_all.txt'))
df_human, raw_data, day_map = preprocess_pipeline(df, normalize=False, transformed=False, aggregation="robust")


# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div(style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'width': '100vw', 'height': '100vh', 'fontSize': '120%'}, children=[
    html.H1("Gene Expression Plotter", style={'textAlign': 'center', 'marginTop': '20px', 'fontSize': '200%'}),

    # Dropdown for gene selection
    dcc.Dropdown(
        id='gene-selector',
        options=[{'label': gene, 'value': gene} for gene in df_human.index],
        multi=True,  # Allows multiple selection
        placeholder="Select genes to plot",
        style={'width': '50%', 'marginBottom': '20px', 'fontSize': '120%'}
    ),
    html.Div(id='color-picker-container', style={'marginBottom': '20px'}),

dcc.Store(id='gene-colors-store', data={}),

# Toggle normalization
html.Button("Log Transform", id="log-transform-btn", n_clicks=0, style={'marginBottom': '20px', 'fontSize': '120%'}),

html.Div(id='log-transform-status', style={'marginBottom': '20px', 'fontSize': '120%'}),

# Choose colors for time series plots
html.Div("Choose colors for time series plots here", style={'marginBottom': '20px', 'fontSize': '120%'}),
     html.Button("Choose colors", id="color-chooser-btn", n_clicks=0, style={'marginBottom': '20px', 'fontSize': '120%'}),
    # Graph display
    dcc.Graph(id='gene-plot', style={'flexGrow': 1, 'width': '90vw', 'height': '600px'}),

    # Buttons for downloading the plot
    html.Div(style={'display': 'flex', 'justifyContent': 'center', 'marginTop': '20px'}, children=[
        html.Button("Download as HTML", id="download-html-btn", style={'marginRight': '10px', 'fontSize': '120%'}),
        dcc.Download(id="download-html"),

        html.Button("Download as SVG", id="download-svg-btn", style={'marginLeft': '10px', 'fontSize': '120%'}),
        dcc.Download(id="download-svg")
    ])
])
def log_transform_data(df_selected):
    """Apply the app's log transformation to selected expression rows."""
    return np.log1p(df_selected)  # Use np.log1p for log(1 + x) to handle zero values gracefully


def get_plot_data(selected_genes, log_transformed):
    """Return aggregated and raw selected data in the same plotting scale."""
    aggregated = df_human.loc[selected_genes]
    raw = raw_data.loc[selected_genes]

    if log_transformed:
        return log_transform_data(aggregated), log_transform_data(raw)
    return aggregated, raw


def raw_points_for_gene(raw_gene_values):
    """Map raw replicate columns to their day-level x/y plotting coordinates."""
    raw_gene_values = raw_gene_values.dropna()
    x_raw = [day_map[column] for column in raw_gene_values.index]
    y_raw = raw_gene_values.values
    return x_raw, y_raw


from dash.dependencies import ALL

@app.callback(
    Output('gene-colors-store', 'data'),
    Input({'type': 'gene-color-picker', 'index': ALL}, 'value'),
    State('gene-selector', 'value'),
    State('gene-colors-store', 'data')
)
def update_gene_colors(picked_colors, selected_genes, gene_colors):
    """Persist color-picker values for the selected genes."""
    if not selected_genes or not picked_colors:
        return gene_colors
    for gene, color in zip(selected_genes, picked_colors):
        if isinstance(color, dict) and 'hex' in color:
            gene_colors[gene] = color['hex']
        elif isinstance(color, str):
            gene_colors[gene] = color
        else:
            # fallback, random
            import random
            gene_colors[gene] = '#%06X' % random.randint(0, 0xFFFFFF)
    return gene_colors



@app.callback(
    [Output('gene-plot', 'figure'),
     Output('log-transform-status', 'children')],
    [Input('gene-selector', 'value'),
     Input('log-transform-btn', 'n_clicks'),
     Input('gene-colors-store', 'data')],
)
def update_plot(selected_genes, log_transform_clicks, gene_colors):
    """Update the expression plot and log-transform status text."""
    if not selected_genes:
        return go.Figure(), ""  # Empty figure

    fig = go.Figure()
    x_values = df_human.columns.astype(int)
    log_transform_clicks = log_transform_clicks or 0
    log_transformed = log_transform_clicks % 2 == 1  # Log transform if clicked an odd number of times
    plot_data, raw_plot_data = get_plot_data(selected_genes, log_transformed)
    log_transform_status = "Data LOG transformed" if log_transformed else "Data is not log transformed"

    for gene, y_values in plot_data.iterrows():
        color = gene_colors.get(gene, None)
        fig.add_trace(go.Scatter(x=x_values, y=y_values.values, mode='lines+markers', name=gene, line=dict(color=color)))
        raw_x, raw_y = raw_points_for_gene(raw_plot_data.loc[gene])
        fig.add_trace(go.Scatter(
            x=raw_x,
            y=raw_y,
            mode='markers',
            name=f"{gene} raw points",
            marker=dict(color='black', size=8, opacity=0.7),
            showlegend=False,
            hovertemplate=(
                "Gene: %{customdata}<br>"
                "Day: %{x}<br>"
                "Raw point: %{y}<extra></extra>"
            ),
            customdata=[gene] * len(raw_x),
        ))

    # Apply a consistent layout every time the figure is updated
    yaxis_title = "Normalized Expression Level" if log_transformed else "Expression Level"

    # After plotting all genes...
    y_means = [y.mean() for _, y in plot_data.iterrows()]
    overall_mean = np.mean(y_means)
    y_max = np.max([y.max() for _, y in plot_data.iterrows()])
    y_min = np.min([y.min() for _, y in plot_data.iterrows()])
    y_mid = (y_max + y_min) / 2

    # Heuristic: place legend on side with less data
    if overall_mean < y_mid:
        # Data is lower, put legend at top
        legend_y = 1
        legend_yanchor = "top"
    else:
        legend_y = 0
        legend_yanchor = "bottom"

    fig.update_layout(
        xaxis_title="Time (days)",
        xaxis_type="category",
        yaxis_title=yaxis_title,
        template="plotly_white",
        margin={'l': 40, 'r': 40, 't': 40, 'b': 40},  # Adjust margins for a cleaner look
        height=600,  # Set a fixed height for better appearance
        showlegend=True,  # Ensure legend is always shown
        font={'size': 24},  # Increase font size for better readability
        legend=dict(
            orientation="v",
            yanchor=legend_yanchor,
            xanchor="right",
            x=1,
            y=legend_y
        )
    )

    return fig, log_transform_status

@app.callback(
    Output("download-html", "data"),
    [Input("download-html-btn", "n_clicks"),
     Input('gene-plot', 'figure')],
    prevent_initial_call=True
)
def download_html(n_clicks, figure):
    """Return the current figure as downloadable HTML when requested."""
    if n_clicks:
        return dcc.send_string(go.Figure(figure).to_html(full_html=True), "gene_expression_plot.html")

@app.callback(
    Output("download-svg", "data"),
    [Input("download-svg-btn", "n_clicks"),
     Input('gene-plot', 'figure')],
    prevent_initial_call=True
)
def download_svg(n_clicks, figure):
    """Return the current figure as downloadable SVG when requested."""
    if n_clicks:
        # Convert the Plotly figure to an SVG string using the kaleido engine
        fig = go.Figure(figure)
        svg_str = pio.to_image(fig, format="svg", engine="kaleido")

        # Send the SVG string as a downloadable file
        return dcc.send_bytes(svg_str, "gene_expression_plot.svg")

@app.callback(
    Output('color-picker-container', 'children'),
    [Input('gene-selector', 'value')],
    [State('gene-colors-store', 'data')]
)
def update_color_pickers(selected_genes, gene_colors):
    """Render one color picker for each selected gene."""
    import random
    if not selected_genes:
        return []
    children = []
    for gene in selected_genes:
        # Use hex string if available, else random color
        hex_color = gene_colors.get(gene, '#%06X' % random.randint(0, 0xFFFFFF))
        # Always pass an object
        color_obj = {'hex': hex_color} if isinstance(hex_color, str) else hex_color
        children.append(
            html.Div([
                html.Label(gene, style={'marginRight': '10px'}),
                daq.ColorPicker(
                    id={'type': 'gene-color-picker', 'index': gene},
                    value=color_obj,
                    size=150,
                )
            ], style={'display': 'flex', 'alignItems': 'center', 'marginBottom': '10px'})
        )
    return children


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
