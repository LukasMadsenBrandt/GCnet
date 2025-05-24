import numpy as np
from gene_analysis_kutsche.data_preprocessing import load_and_preprocess_data as load_and_preprocess_kutsche
from gene_analysis_kutsche.data_filtering import preprocess_pipeline as preprocess_pipeline
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.io as pio
import pandas as pd
import os
from statsmodels.tsa.api import VAR

# Load and filter data
df = load_and_preprocess_kutsche(os.path.join('Data', 'Kutsche', 'genes_all.txt'))
df_human, raw_data, day_map = preprocess_pipeline(df, logged=False)


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

    # Toggle normalization
    html.Button("Log Transform", id="log-transform-btn", n_clicks=0, style={'marginBottom': '20px', 'fontSize': '120%'}),
    html.Div(id='log-transform-status', style={'marginBottom': '20px', 'fontSize': '120%'}),

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
    return df_selected.apply(np.log)

@app.callback(
    [Output('gene-plot', 'figure'),
     Output('log-transform-status', 'children')],
    [Input('gene-selector', 'value'),
     Input('log-transform-btn', 'n_clicks')],
    [State('log-transform-btn', 'n_clicks_timestamp')]
)
def update_plot(selected_genes, n_clicks, n_clicks_timestamp):
    if not selected_genes:
        return go.Figure(), ""  # Empty figure

    fig = go.Figure()
    x_values = df_human.columns.astype(int)
    log_transformed = n_clicks % 2 == 1  # Log transform if the button has been clicked an odd number of times

    # Extract selected gene data
    df_selected = df_human.loc[selected_genes]

    if log_transformed:
        log_transformed_data = log_transform_data(df_selected)
        log_transform_status = "Data log transformed"
    else:
        log_transformed_data = df_selected
        log_transform_status = "Data is not log transformed"

    # Plot the (log transformed or raw) data
    for gene, y_values in log_transformed_data.iterrows():
        fig.add_trace(go.Scatter(x=x_values, y=y_values.values, mode='lines+markers', name=gene))

    # Apply a consistent layout every time the figure is updated
    yaxis_title = "Log Expression Level" if log_transformed else "Expression Level"
    fig.update_layout(
        title="Gene Expression Over Time",
        xaxis_title="Time (days)",
        xaxis_type="category",
        yaxis_title=yaxis_title,
        template="plotly_white",
        margin={'l': 40, 'r': 40, 't': 40, 'b': 40},  # Adjust margins for a cleaner look
        height=600,  # Set a fixed height for better appearance
        showlegend=True,  # Ensure legend is always shown
        font={'size': 16}  # Increase font size for better readability
    )

    return fig, log_transform_status

@app.callback(
    Output("download-html", "data"),
    [Input("download-html-btn", "n_clicks"),
     Input('gene-plot', 'figure')],
    prevent_initial_call=True
)
def download_html(n_clicks, figure):
    if n_clicks:
        return dcc.send_string(go.Figure(figure).to_html(full_html=True), "gene_expression_plot.html")

@app.callback(
    Output("download-svg", "data"),
    [Input("download-svg-btn", "n_clicks"),
     Input('gene-plot', 'figure')],
    prevent_initial_call=True
)
def download_svg(n_clicks, figure):
    if n_clicks:
        # Convert the Plotly figure to an SVG string using the kaleido engine
        fig = go.Figure(figure)
        svg_str = pio.to_image(fig, format="svg", engine="kaleido")

        # Send the SVG string as a downloadable file
        return dcc.send_bytes(svg_str, "gene_expression_plot.svg")

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)

