import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit
import plotly.io as pio

# Updated data: Number of Louvain runs and corresponding unique genes observed with ZEB2
runs = np.array([
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    15, 20, 25, 30, 35, 40, 45, 50,
    60, 70, 80, 90, 100,
    125, 150, 175, 200, 225, 250,
    300, 350, 400, 450, 500,
    600, 700, 800, 900, 1000,
    1250, 1500, 1750, 2000, 2250, 2500,
    3000, 3500, 4000, 4500, 5000,
    6000, 7000, 8000, 9000, 10000
])

unique_genes = np.array([
    64, 121, 150, 214, 233, 257, 313, 373, 392, 424,
    606, 718, 834, 877, 985, 1016, 1064, 1095,
    1147, 1177, 1273, 1326, 1366,
    1472, 1564, 1667, 1702, 1733, 1739,
    1774, 1790, 1814, 1814, 1821,
    1826, 1834, 1838, 1838, 1843,
    1845, 1854, 1854, 1857, 1857, 1857,
    1866, 1866, 1866, 1866, 1866,
    1866, 1866, 1866, 1866, 1866
])

total_genes = 1866

# Define the Michaelis-Menten function
def michaelis_menten(x, L, k):
    return L * x / (x + k)

# Fit the Michaelis-Menten model
initial_guess = [total_genes, 10]
popt, pcov = curve_fit(michaelis_menten, runs, unique_genes, p0=initial_guess)
L_fit, k_fit = popt
print(f"Fitted parameters: L = {L_fit:.2f}, k = {k_fit:.2f}")

# Generate smooth curve for plotting
x_fit = np.linspace(0, runs[-1], 1000)
y_fit = michaelis_menten(x_fit, L_fit, k_fit)

# Create interactive Plotly figure
fig = go.Figure()

# Add observed data points as markers
fig.add_trace(go.Scatter(
    x=runs,
    y=unique_genes,
    mode='markers+lines',
    name='Observed Data',
    marker=dict(size=8, color='blue'),
    hovertemplate = "Louvain Runs: %{x}<br>Unique Genes: %{y}<extra></extra>"
))

# Add the fitted Michaelis-Menten curve as a line
fig.add_trace(go.Scatter(
    x=x_fit,
    y=y_fit,
    mode='lines',
    name=f'Fit: L = {L_fit:.1f}, k = {k_fit:.1f}',
    line=dict(color='red'),
    hovertemplate = "Louvain Runs: %{x:.0f}<br>Fitted Unique Genes: %{y:.1f}<extra></extra>"
))

# Update layout: remove x-axis tick labels if desired, and add titles and grid
fig.update_layout(
    title="Michaelis-Menten Regression on Unique Genes vs. Louvain Runs",
    xaxis_title="Number of Louvain Runs",
    yaxis_title="Unique Genes in Same Community as ZEB2",
    xaxis=dict(showticklabels=True),
    template="plotly_white"
)

# Save interactive plot as HTML
html_filename = "unique_genes_michaelis_menten_interactive.html"
pio.write_html(fig, file=html_filename, auto_open=False)
print(f"Interactive plot saved as {html_filename}")

# Optionally, display the interactive figure in your environment:
fig.show()
