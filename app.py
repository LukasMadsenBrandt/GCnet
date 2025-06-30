import subprocess
import concurrent.futures
import matplotlib
matplotlib.use('Agg')
import shutil
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
import graphviz
import base64
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate

import dash_bootstrap_components as dbc
from plotly.subplots import make_subplots
import plotly.io as pio

# Community detection libraries
from community import community_louvain
from networkx.algorithms.community.centrality import girvan_newman


from functools import lru_cache
import hashlib
import pickle
import html as html_escape
import os
import pandas as pd
from collections import Counter
import warnings



# Importing functions from both folders
from gene_analysis_benito.granger_causality import perform_granger_causality_tests as perform_gc_benito
from gene_analysis_benito.granger_causality import filter_gene_pairs as filter_gene_pairs_benito
from gene_analysis_benito.granger_causality import collect_significant_edges as collect_significant_edges_benito
from gene_analysis_benito.data_preprocessing import filter_data_proximity_based_weights as filter_proximity_benito
from gene_analysis_benito.data_preprocessing import filter_data_arithmetic_mean as filter_mean_benito
from gene_analysis_benito.data_preprocessing import filter_data_median as filter_median_benito
from gene_analysis_benito.data_filtering import filter_data as mapper_benito

from gene_analysis_kutsche.granger_causality import perform_granger_causality_tests as perform_gc_kutsche
from gene_analysis_kutsche.granger_causality import filter_gene_pairs as filter_gene_pairs_kutsche
from gene_analysis_kutsche.granger_causality import collect_significant_edges as collect_significant_edges_kutsche
from gene_analysis_kutsche.data_preprocessing import load_and_preprocess_data as load_and_preprocess_kutsche
from gene_analysis_kutsche.data_filtering import preprocess_pipeline as preprocess_pipeline


#warnings.filterwarnings("ignore", message="'linear' x-axis tick spacing not even")
warnings.filterwarnings("ignore")


# Config
pvalue_global = 0.0004
genelist_global = ['ZEB2']
weighted_edges = False
debugging = True
# Create cache directory if it doesn't exist

cache_dir = os.path.join(os.getcwd(), 'cache')
plots_dir = os.path.join(os.getcwd(), 'plots')
dirs = [cache_dir, plots_dir]

for directory in dirs:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Debugging utilities
def debug_print(*args):
    if debugging:
        print(" ".join(map(str, args)))


def debug_print_edges(edges, message):
    debug_print(message)
    for edge in edges:
        debug_print(edge)

# Additional debug functions
def debug_print_nodes(graph, message):
    debug_print(message)
    for node in graph.nodes():
        debug_print(node)

def debug_print_graph_details(graph, message):
    debug_print(message)
    debug_print(f"Nodes: {list(graph.nodes())}")
    debug_print(f"Edges: {list(graph.edges(data=True))}")

def get_cache_filename(dataset, method):
    filename = os.path.join(cache_dir, f'{dataset}_data_{method}.pkl')
    debug_print(f"Cache file: {filename}")  # Debug statement
    return filename

def save_cache(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    debug_print(f"Saved cache to {filename}")  # Debug statement

def load_cache(filename):
    if os.path.exists(filename):
        debug_print(f"Loading cache from {filename}")  # Debug statement
        with open(filename, 'rb') as f:
            return pickle.load(f)
    debug_print(f"No cache found for {filename}")  # Debug statement
    return None

def clear_directory(dir):
    for filename in os.listdir(dir):
        file_path = os.path.join(dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            debug_print(f'Failed to delete {file_path}. Reason: {e}')


# Create a directed graph based on significant edges
def create_network(significant_edges):
    G = nx.DiGraph()
    for edge in significant_edges:
        if len(edge) == 3:  # For benito and kutsche datasets
            (source, lag), (target, _), p_value = edge
            G.add_edge(source, target, lag=lag, p_value=p_value)
        elif len(edge) == 4:  # For intersection dataset
            (source, lag), (target, _), kutsche_p_value, benito_p_value = edge
            avg_p_value = (kutsche_p_value + benito_p_value) / 2
            G.add_edge(source, target, lag=lag, kutsche_p_value=kutsche_p_value, benito_p_value=benito_p_value, p_value=avg_p_value)
        else:
            debug_print(f"Something went wrong with edge: {edge}")
            raise ValueError("Invalid edge format")
    return G


# Hash the significant edges for caching
def hash_significant_edges(significant_edges):
    return hashlib.md5(pickle.dumps(significant_edges)).hexdigest()

# Convert partition to tuple for caching
def partition_to_tuple(partition):
    return tuple(sorted(partition.items()))

# Convert tuple back to partition
def tuple_to_partition(t):
    return dict(t)

# Apply community detection to a graph
@lru_cache(maxsize=1)
def cached_apply_community_detection(graph_pickle, method, num_communities):
    G = pickle.loads(graph_pickle)  # Deserialize the graph
    return apply_community_detection(G, method, num_communities)

# Apply Girvan-Newman community detection algorithm to a directed graph
def girvan_newman_community_detection(G, num_communities=2):
    """
    Apply Girvan-Newman community detection algorithm to a directed graph.
    """
    if G.number_of_nodes() == 0:
        debug_print("Graph is empty, no community detection possible.")
        return {}

    # Perform Girvan-Newman community detection
    communities_generator = girvan_newman(G)

    # Iterate through the generator to get the required number of communities
    communities = []
    try:
        for communities in communities_generator:
            if len(communities) >= num_communities:
                break
    except StopIteration:
        debug_print("Could not find the specified number of communities.")
        return {}

    if len(communities) < num_communities:
        debug_print("The specified number of communities cannot be obtained.")
        return {}

    # Create a partition dictionary
    partition = {}
    for community_number, community in enumerate(communities):
        for node in community:
            partition[node] = community_number

    return partition


from sklearn.cluster import AgglomerativeClustering

def run_louvain_once(G, seed):
    partition = community_louvain.best_partition(G.to_undirected(), random_state=seed)
    return partition

def run_multiple_louvain_parallel(G, n_runs=100):
    partitions = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_louvain_once, G, seed) for seed in range(n_runs)]
        for future in concurrent.futures.as_completed(futures):
            partitions.append(future.result())
    debug_print(f"Consensus detection complete: {len(partitions)} partitions run (expected {n_runs}).")
    return partitions


def build_coassociation_matrix(G, partitions):
    nodes = list(G.nodes())
    n = len(nodes)
    coassoc = np.zeros((n, n))
    for part in partitions:
        for i in range(n):
            for j in range(i, n):
                if part.get(nodes[i]) == part.get(nodes[j]):
                    coassoc[i, j] += 1
                    if i != j:
                        coassoc[j, i] += 1
    coassoc /= len(partitions)
    return nodes, coassoc

def consensus_partition(G, n_runs=20, n_clusters=None, gene_of_interest=None, plot = None):
    partitions = run_multiple_louvain_parallel(G, n_runs)
    nodes, coassoc = build_coassociation_matrix(G, partitions)
    
    #plot = True
    # If n_clusters is not provided, compute it as the average number of communities over all partitions.
    if n_clusters is None:
        total_communities = sum(len(set(partition.values())) for partition in partitions)
        avg_communities = total_communities / len(partitions)
        n_clusters = int(round(avg_communities))
    
    # Convert coassociation to a distance matrix (1 - coassociation)
    distance_matrix = 1 - coassoc
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(distance_matrix)
    consensus = {node: labels[i] for i, node in enumerate(nodes)}
    
    # Compute the union of all genes that have ever been in the same community as gene_of_interest
    union_genes = set()
    if gene_of_interest is not None:
        for partition in partitions:
            if gene_of_interest in partition:
                current_comm = partition[gene_of_interest]
                union_genes.update([node for node, comm in partition.items() if comm == current_comm])
                #### DEBUGGING ####
                #debug_print(f"Partition: {partition}")
                #debug_print(f"Gene of interest {gene_of_interest} is in community {current_comm}")
                #debug_print(f"Unique gene: {union_genes}")
                #debug_print(f"Number of unique genes: {len(union_genes)}")
        # Plot the consensus matrix, I.E the freaquencies of two genes being in the same community
        # Only plot the matrix with the pair of the gene of interest and the rest of the genes
        if plot != None:
            plot_coassociation_for_gene(coassoc, nodes, gene_of_interest, n_runs, plots_dir)
        num_genes_same_comm = len(union_genes)
    else:
        num_genes_same_comm = None

    return consensus, coassoc, partitions, num_genes_same_comm

import plotly.express as px
import plotly.io as pio

def plot_coassociation_for_gene(coassoc, nodes, gene_of_interest, n_runs, save_dir):
    """
    Plot an interactive scatter plot showing the coassociation frequency for each gene 
    with respect to the gene_of_interest, based on the coassociation matrix.
    
    Parameters:
        coassoc (np.ndarray): The coassociation matrix (assumed symmetric).
        nodes (list): List of gene names corresponding to the rows/columns of coassoc.
        gene_of_interest (str): The gene to inspect (e.g. "ZEB2").
        n_runs (int): Number of consensus runs (used in the filename).
        save_dir (str): Directory to save the plot.

    Returns:
        fig (plotly.graph_objects.Figure): The interactive figure.
    """
    gene_of_interest = "ZEB2"
    # Find the index of the gene of interest in the nodes list.
    try:
        idx = nodes.index(gene_of_interest)
    except ValueError:
        raise ValueError(f"Gene {gene_of_interest} not found in the nodes list.")

    # Extract the row corresponding to the gene of interest.
    frequencies = coassoc[idx, :]

    # Create a DataFrame for plotting.
    import pandas as pd
    df = pd.DataFrame({
        'Gene': nodes,
        'Coassociation Frequency': frequencies,
        'Index': list(range(len(nodes)))
    })
    # Remove the gene of interest itself.
    df = df[df['Gene'] != gene_of_interest]
    
    # Create an interactive scatter plot using Plotly Express.
    fig = px.scatter(
        df,
        x='Index',
        y='Coassociation Frequency',
        hover_data={'Gene': True, 'Coassociation Frequency': ':.4f'},
        title=f'Coassociation Frequencies for Genes with {gene_of_interest}',
    )
    
    # Remove x-axis tick labels since we don't need gene names there.
    fig.update_layout(
        xaxis=dict(
            showticklabels=False,
            title="",
        ),
        yaxis_title="Coassociation Frequency",
    )

    # Save the interactive plot as an HTML file.
    html_filename = f"coassoc_{gene_of_interest}_n_runs_{n_runs}.html"
    html_path = os.path.join(save_dir, html_filename)
    pio.write_html(fig, file=html_path, auto_open=False)
    debug_print(f"Interactive coassociation plot saved to {html_path}")

    # Additionally, save as a static image (PNG) if needed.
    png_filename = f"coassoc_{gene_of_interest}_n_runs_{n_runs}.png"
    png_path = os.path.join(save_dir, png_filename)
    try:
        fig.write_image(png_path)
        debug_print(f"Static coassociation plot saved to {png_path}")
    except Exception as e:
        debug_print(f"Error saving static image: {e}")

    return fig



def apply_community_detection(G, method='louvain', num_communities=None):
    debug_print(f"Applying community detection method: {method}, num_communities: {num_communities}")
    if G.number_of_nodes() == 0:
        debug_print("Graph is empty, no community detection possible.")
        return {}

    try:
        if method == 'louvain':
            partition = community_louvain.best_partition(G.to_undirected(), random_state=42)
        elif method == 'girvan_newman':
            if num_communities is None:
                louvain_partition = community_louvain.best_partition(G.to_undirected(), random_state=42)
                num_communities = len(set(louvain_partition.values()))
            partition = girvan_newman_community_detection(G, num_communities)
        else:
            raise ValueError(f"Unsupported community detection method: {method}")

        if num_communities and len(set(partition.values())) > num_communities:
            partition = reduce_communities(partition, num_communities)
        
        debug_print(f"Number of detected communities: {len(set(partition.values()))}")

        debug_print("Community detection successful.")
        return partition
    except Exception as e:
        debug_print(f"Error during community detection: {e}")
        return {}



# Reduce the number of communities to the specified number
def reduce_communities(partition, num_communities):
    community_counts = Counter(partition.values())
    most_common_communities = [community for community, _ in community_counts.most_common(num_communities)]
    
    new_partition = {}
    for node, community in partition.items():
        if community in most_common_communities:
            new_partition[node] = community
        else:
            new_partition[node] = min(most_common_communities)
    
    return new_partition


# Assign colors to communities
@lru_cache(maxsize=32)
def cached_assign_colors(partition_tuple):
    partition = tuple_to_partition(partition_tuple)
    return assign_colors(partition)

# Modified assign_colors function to label communities numerically
def assign_colors(partition):
    debug_print(f"Assigning colors to communities: {set(partition.values())}")
    cmap = plt.get_cmap('tab20')
    community_colors = {}
    unique_communities = sorted(set(partition.values()))  # Sort to maintain consistency

    for idx, community in enumerate(unique_communities):
        color = cmap(idx % cmap.N)
        color_hex = '#%02x%02x%02x' % tuple(int(255 * c) for c in color[:3])
        community_colors[community] = (color_hex, str(idx + 1))  # Label communities as 1, 2, 3, ...

    debug_print(f"Assigned community colors: {community_colors}")
    return community_colors



# Normalize values for visualization
def normalize(values, min_size=0.1, max_size=2.0):
    if not values:
        return [min_size] * len(values)

    min_val, max_val = min(values), max(values)
    range_val = max_val - min_val
    if range_val == 0:
        return [min_size for _ in values]

    return [min_size + (max_size - min_size) * (v - min_val) / range_val for v in values]

# Create Graphviz DOT representation
def create_graphviz_dot(G, partition, community_colors, highlight_node=None, layout="dot", graph_attr={}, weighted_edges=False, simple_layout=False):
    dot = graphviz.Digraph(engine=layout, format='svg', graph_attr=graph_attr)

    dot.attr(tooltip='')

    # If the graph is empty, return an empty graph
    if not G.nodes:
        return dot

    outdegrees = [G.out_degree(node) for node in G.nodes()]
    norm_outdegrees = normalize(outdegrees, min_size=1, max_size=2)

    for idx, node in enumerate(G.nodes()):
        community = partition.get(node, None)
        color_hex, color_name = community_colors.get(community, ("#d3d3d3", "gray"))
        if weighted_edges:
            size = str(norm_outdegrees[idx])
        else:
            size = str(1)

        out_edges = G.out_edges(node, data=True)
        out_edges_info = []
        for _, target, data in out_edges:
            kutsche_p_value = data.get('kutsche_p_value', None)
            benito_p_value = data.get('benito_p_value', None)
            if kutsche_p_value is not None and benito_p_value is not None:
                out_edges_info.append(f"{target}: ({kutsche_p_value:.6f})({benito_p_value:.6f})")
            else:
                out_edges_info.append(f"{target}: ({data['p_value']:.6f})")
        
        out_edges_info = ", ".join(out_edges_info)
        if any('kutsche_p_value' in data and 'benito_p_value' in data for _, _, data in out_edges):
            hover_text = html_escape.escape(
                f'{node} may granger cause {G.out_degree(node)} gene(s), p-values are formatted in the following way: Gene: (Kutsche p-value)(Benito-Kwiecinski p-value) : \n{out_edges_info}'
            )
        else:
            hover_text = html_escape.escape(
                f'{node} may granger cause {G.out_degree(node)} gene(s), p-values are formatted in the following way: Gene: (p-value) \n{out_edges_info}'
            )

        node_style = 'filled'
        node_penwidth = '3' if node == highlight_node else '1'

        if simple_layout:
            # For simple layout, use a smaller size and different style
            dot.node(node, label=node, shape='plaintext', style=node_style, fillcolor='white', color='black', penwidth='1', tooltip=hover_text, width=size, height=size, fixedsize='true')
        else:
            dot.node(node, label=node, shape='circle', style=node_style, fillcolor=color_hex, color='black', penwidth=node_penwidth, tooltip=hover_text, width=size, height=size, fixedsize='true')

    if not G.edges:
        return dot

    p_values = [G[source][target].get('p_value', 1.0) for source, target in G.edges()]
    if weighted_edges == True:
        norm_p_values = normalize([-np.log10(p) for p in p_values], min_size=1, max_size=10)
    else: 
        norm_p_values = normalize([1 for p in p_values], min_size=3, max_size=3)

    for idx, (source, target) in enumerate(G.edges()):
        lag = G[source][target]['lag']
        p_value = G[source][target]['p_value']
        kutsche_p_value = G[source][target].get('kutsche_p_value', None)
        benito_p_value = G[source][target].get('benito_p_value', None)

        weight = str(norm_p_values[idx])
        color_hex, _ = community_colors.get(partition.get(source), ("#d3d3d3", "gray"))
        if kutsche_p_value is not None and benito_p_value is not None:
            hover_text = html_escape.escape(
                f'{source} may Granger Cause {target} at lag {lag} with a p-value of {kutsche_p_value:.6f}(Kutsche) and a p-value of {benito_p_value:.6f}(Benito-Kwiecinski)'
            )
        else:
            hover_text = html_escape.escape(
                f'{source} may Granger Cause {target} at lag {lag} with a p-value of {p_value:.6f}'
            )

        if highlight_node and source == highlight_node:
            color_hex, weight = 'black', str(norm_p_values[idx] + 3)

        if simple_layout:
            dot.edge(source, target, color='black', penwidth='1', tooltip=hover_text)
        else:
            dot.edge(source, target, color=color_hex, penwidth=weight, tooltip=hover_text)

    return dot



# Function to compare datasets and find intersections
def compare_datasets(kutsche, benito):
    dict1 = {(row['Gene1'], row['Gene2'], row['Lag']): row['P_Value'] for index, row in kutsche.iterrows()}
    dict2 = {(row['Gene1'], row['Gene2'], row['Lag']): row['P_Value'] for index, row in benito.iterrows()}

    common_edges = []
    for key in dict1:
        if key in dict2:
            gene1, gene2, lag = key
            kutsche_p_value = dict1[key]
            benito_p_value = dict2[key]
            common_edges.append(((gene1, lag), (gene2, '_'), kutsche_p_value, benito_p_value))

    return common_edges

def update_graph_function(significant_edges, selected_communities, search_value, layout,
                          community_detection_method="louvain", num_communities=None,
                          partition_override=None):
    # Create network from edges
    G = create_network(significant_edges)
    
    if G.number_of_nodes() == 0:
        return None, [], "Graph is empty."

    # Use the provided partition if it exists.
    if partition_override is not None:
        partition = partition_override
    else:
        graph_pickle = pickle.dumps(G)
        partition = cached_apply_community_detection(
            graph_pickle,
            method=community_detection_method,
            num_communities=num_communities if community_detection_method != 'louvain' else None
        )

    # Now, assign colors using the partition.
    partition_tuple = partition_to_tuple(partition)
    community_colors = cached_assign_colors(partition_tuple)
    
    # Apply community filtering if the user selected a subset.
    if selected_communities:
        nodes_to_keep = {node for node, comm in partition.items() if comm in selected_communities}
        G = G.subgraph(nodes_to_keep).copy()

    # Error handling for search value.
    if search_value and search_value not in G.nodes():
        error_message = f"Gene '{search_value}' not found in the graph."
        debug_print(error_message)
    else:
        error_message = ""

    # Build checklist options using the assigned colors.
    community_options = [
        {'label': html.Span(f'Community {comm}', style={'color': community_colors.get(comm, ("#000000",))[0]}),
         'value': comm} for comm in set(partition.values())
    ]
    
    # Create the Graphviz DOT representation.
    dot = create_graphviz_dot(G, partition, community_colors, highlight_node=search_value, layout=layout)
    return dot, community_options, error_message




# HTML template for embedding Graphviz output with zoom and pan
html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>Network Graph</title>
    <style>
        body, html {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
            font-family: Arial, sans-serif;
        }}
        #graph {{
            width: 100%;
            height: 100%;
            border: none;
            overflow: hidden;
        }}
        .button-container {{
            position: absolute;
            top: 10px;
            right: 10px;
            z-index: 1000;
            display: flex;
            flex-direction: column;
        }}
        .zoom-button {{
            display: inline-block;
            padding: 10px;
            margin: 5px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
            font-size: 14px;
            border-radius: 5px;
        }}
        .zoom-button:hover {{
            background-color: #45a049;
        }}
    </style>
</head>
<body>
    <div id="graph">{graph}</div>
    <div class="button-container">
        <button class="zoom-button" id="zoom-in">Zoom In</button>
        <button class="zoom-button" id="zoom-out">Zoom Out</button>
    </div>
    <script>
        const svg = document.querySelector('#graph svg');
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', '100%');
        svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

        function setInitialViewBox() {{
            const bbox = svg.getBBox();
            const padding = 20; // Add padding to fit within the window
            const scale = 1.02; // Scale down slightly to fit the height
            const width = bbox.width + 2 * padding;
            const height = (bbox.height + 2 * padding) * scale;
            const viewBox = [bbox.x - padding, bbox.y - padding, width, height].join(' ');
            svg.setAttribute('viewBox', viewBox);
        }}

        setInitialViewBox();

        const zoomFactor = 1.2;
        document.getElementById('zoom-in').addEventListener('click', () => {{
            const viewBox = svg.getAttribute('viewBox').split(' ').map(Number);
            viewBox[2] /= zoomFactor;
            viewBox[3] /= zoomFactor;
            svg.setAttribute('viewBox', viewBox.join(' '));
        }});

        document.getElementById('zoom-out').addEventListener('click', () => {{
            const viewBox = svg.getAttribute('viewBox').split(' ').map(Number);
            viewBox[2] *= zoomFactor;
            viewBox[3] *= zoomFactor;
            svg.setAttribute('viewBox', viewBox.join(' '));
        }});

        let isPanning = false;
        let startX, startY;
        svg.addEventListener('mousedown', (event) => {{
            isPanning = true;
            startX = event.clientX;
            startY = event.clientY;
        }});
        svg.addEventListener('mousemove', (event) => {{
            if (isPanning) {{
                const dx = startX - event.clientX;
                const dy = startY - event.clientY;
                const viewBox = svg.getAttribute('viewBox').split(' ').map(Number);
                svg.setAttribute('viewBox', `${{viewBox[0] + dx}} ${{viewBox[1] + dy}} ${{viewBox[2]}} ${{viewBox[3]}}`);
                startX = event.clientX;
                startY = event.clientY;
            }}
        }});
        svg.addEventListener('mouseup', () => {{
            isPanning = false;
        }});
        svg.addEventListener('mouseleave', () => {{
            isPanning = false;
        }});
    </script>
</body>
</html>
"""


# Define layout options
layout_options = [
    {'label': 'DOT (Hierarchical)', 'value': 'dot', 'description': 'Hierarchical or layered drawings of directed graphs.'},
    {'label': 'FDP (Force-Directed Placement)', 'value': 'fdp', 'description': 'Force-Directed Placement.'},
    {'label': 'SFDP (Scalable FDP) Mac & Linux', 'value': 'sfdp', 'description': 'Scalable Force-Directed Placement.'},
    {'label': 'CIRCO (Circular)', 'value': 'circo', 'description': 'Circular layout.'},
]

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# Define the layout of the app
app.layout = html.Div([
    dcc.Store(id='current-dataset', data=None),
    dcc.Store(id='current-summarization-technique', data=None),
    dcc.Store(id='toggle-state', data=True),
    dcc.Store(id='df-store'),  # Store for the DataFrame
    html.Div([
        # Controls
        html.Label('Dataset:', style={'color': 'white'}),
        dcc.Dropdown(
            id='dataset-dropdown',
            options=[
                {'label': '47-Benito-Kwiecinski', 'value': 'benito'},
                {'label': '47-Kutsche', 'value': 'kutsche'},
                {'label': 'Kutsche', 'value': 'large_kutsche' },
                {'label': 'Kutsche Top 5%', 'value': 'large_kutsche_top5%' },
                {'label': 'Benito Human', 'value': 'large_benito_human' },
                {'label': 'Benito Gorilla', 'value': 'large_benito_gorilla' },
                {'label': 'Intersection', 'value': 'intersection'}
            ],
            style={'backgroundColor': 'white', 'color': 'black', 'marginBottom': '20px'}
        ),
        html.Label('Summarization Technique:', style={'color': 'white'}),
        dcc.Dropdown(
            id='summarization-technique-dropdown',
            options=[
                {'label': 'Median Proximity Weights', 'value': 'proximity'},
                {'label': 'Median', 'value': 'median'},
                {'label': 'Arithmetic Mean', 'value': 'mean'}
            ],
            value='proximity',
            style={'backgroundColor': 'white', 'color': 'black', 'marginBottom': '20px'}
        ),
        dbc.Button("Send", id="send-button", color="primary", style={'marginTop': '10px', 'marginBottom': '20px'}),
        dcc.Loading(
            id="loading-send",
            type="default",
            children=[
                html.Div(id='instructions', children='Please select options and press "Send" to generate the graph.', style={'marginTop': '20px', 'color': 'red'}),
                html.Div(id='community-checklist-container-unique'),
            ]
        ),
        html.Label('P-value threshold:', style={'color': 'white'}),
        html.Div([
            dcc.Slider(
                id='p-threshold-slider',
                min=0.01,
                max=0.1,
                step=0.01,
                value=0.05,
                marks={i / 100: str(i / 100) for i in range(1, 11)},
                tooltip={"placement": "bottom", "always_visible": True},
            ),
            html.Div([
                dcc.Input(
                    id='p-threshold-input',
                    type='number',
                    min=0.000001,
                    max=1,
                    step=0.000001,
                    value=0.05,
                    style={'width': '100px', 'backgroundColor': 'white', 'color': 'black'}
                ),
                dbc.Button("Apply", id="apply-button", color="primary", size="sm", style={'marginLeft': '10px'})
            ], style={'display': 'flex', 'alignItems': 'center'})
        ]),
        html.Label('Search for a gene:', style={'marginTop': '20px', 'color': 'white'}),
        dcc.Input(id='search-bar', type='text', placeholder='Search for a gene...', debounce=True, value='', style={'width': '100%', 'marginBottom': '20px', 'backgroundColor': 'white', 'color': 'black'}),
        dbc.Button("Show Plots", id="show-plots-button", color="primary", style={'marginTop': '10px'}),
        html.Label('Graph Layout:', style={'marginTop': '20px', 'color': 'white'}),
        dcc.Dropdown(
            id='layout-dropdown',
            options=[{'label': opt['label'], 'value': opt['value']} for opt in layout_options],
            value='dot',
            style={'backgroundColor': 'white', 'color': 'black', 'marginBottom': '20px'}
        ),
        
        html.Label('Community Detection Method:', style={'color': 'white'}),
        dcc.Dropdown(
            id='community-detection-method-dropdown',
            options=[
                {'label': 'Louvain Undirected edges', 'value': 'louvain'},
                {'label': 'Girvan-Newman Directed edges', 'value': 'girvan_newman'},
                {'label': 'Consensus (multiple Louvain runs)', 'value': 'consensus'}
            ],
            value='louvain',
            style={'backgroundColor': 'white', 'color': 'black', 'marginBottom': '20px'}
        ),
        html.Div(id='num-of-consensus-runs-container', children=[
            html.Label('Number of Consensus Runs:', style={'color': 'white', 'display': 'inline-block', 'marginRight': '10px'}),
            dcc.Input(id='num-of-consensus-runs', type='number', min=1, step=1, max=10000000, value=100, style={'marginBottom': '20px', 'display': 'inline-block'}),
        ], style={'display': 'none'}),
        html.Div(id='num-of-communities-container', children=[
            html.Label('Number of Communities:', style={'color': 'white', 'display': 'inline-block', 'marginRight': '10px'}),
            dcc.Input(id='num-of-communities', type='number', min=1, step=1, max=20, style={'marginBottom': '20px', 'display': 'inline-block'}),
        ], style={'display': 'none'}),
        dbc.Button("Toggle All", id="toggle-all-button", color="primary", style={'marginTop': '10px'}),
        dbc.Button("Apply Communities", id="apply-communities-button", color="primary", style={'marginTop': '10px'}),
        # Hidden store to keep track of the applied community selection.
        dcc.Store(id='heavy-network-store', data={}),
        html.Label('Select Communities:', style={'marginTop': '20px', 'color': 'white'}),
        html.Div(id='community-checklist-container-2', children=[
            dbc.Checklist(
                id='community-checklist',
                options=[],  # Options will be set in callback
                value=[],    # All values will be selected in the callback
                inline=True,
                style={'color': 'white'}
            ),
        ], style={'maxHeight': '200px', 'overflowY': 'scroll'}),
        
        html.Div(id='node-edge-info', style={'marginTop': '20px', 'color': 'white'}),
        html.Div(id='selections-output', style={'color': 'red'}),
        dbc.Button("Export Graph as HTML", id="export-html-button", color="success", style={'marginTop': '20px'}),
    ], style={'position': 'fixed', 'top': '10px', 'left': '10px', 'width': '300px', 'zIndex': 1000, 'backgroundColor': 'rgba(0,0,0,0.65)', 'padding': '10px', 'borderRadius': '10px'}),
    html.Div([
        html.Div(id='store-output'),
        dcc.Loading(
            id="loading-network-graph",
            type="default",
            children=[
                html.Iframe(id='network-graph', style={'width': '100%', 'height': '100vh', 'border': 'none', 'display': 'block'}),
            ]
        ),
        dcc.Loading(
            id="loading-expression-plots",
            type="default",
            children=[
                html.Div(id='expression-plots')  # Container for the expression plots
            ]
        ),
        dcc.Download(id='download-graph-html')  # Add Download component
    ])    
], style={'padding': '20px', 'position': 'relative'})



# Apply custom CSS styles
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body, html {
                margin: 0;
                padding: 0;
                width: 100%;
                height: 100%;
                overflow: hidden;
                font-family: Arial, sans-serif;
            }
            .Select-menu-outer {
                background-color: black !important;
                color: white !important;
            }
            .Select-arrow-zone, .Select-clear-zone {
                color: white !important;
            }
            .Select--multi .Select-value {
                background-color: black !important;
                color: white !important;
            }
            iframe {
                display: block !important;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

@app.callback(
    Output('num-of-consensus-runs-container', 'style'),
    Input('community-detection-method-dropdown', 'value')
)
def show_hide_consensus_runs(method):
    if method == 'consensus':
        return {'display': 'block'}
    return {'display': 'none'}


# Function that updates that recomputes or load the data from cache, whenever the user presses the "Send" button
@app.callback(
    [Output('community-checklist', 'options'),
     Output('community-checklist', 'value', allow_duplicate=True),
     Output('selections-output', 'children', allow_duplicate=True),
     Output('instructions', 'style'),
     Output('instructions', 'children'),
     Output('df-store', 'data'),  # Store the DataFrame
     Output('current-dataset', 'data'),  # Store the current dataset
     Output('current-summarization-technique', 'data'),  # Store the current summarization technique
     Output('search-bar', 'value'),  # Clear the search bar
     Output('num-of-communities', 'value')],  # Set the number of communities
    [Input('send-button', 'n_clicks')],
    [State('dataset-dropdown', 'value'),
     State('summarization-technique-dropdown', 'value'),
     State('community-detection-method-dropdown', 'value'),
     State('toggle-state', 'data')],
    prevent_initial_call=True
)
def send_selections(n_clicks, dataset, summarization_technique, community_detection_method, toggle_state):
    if n_clicks is None:
        return [], [], "", {'display': 'block'}, 'Please select options and press "Send" to generate the graph.', None, None, None, '', 2

    if not dataset or not summarization_technique:
        return [], [], "", {'display': 'block'}, 'Both dataset and summarization technique must be selected.', None, None, None, '', 2

    global benito_data, kutsche_data

    data_dict = None
    filter_function = None
    data_human = None

    debug_print(f"Button clicked. Dataset: {dataset}, Summarization Technique: {summarization_technique}")

    if dataset == 'benito':
        if summarization_technique == 'proximity':
            filter_function = filter_proximity_benito
        elif summarization_technique == 'mean':
            filter_function = filter_mean_benito
        elif summarization_technique == 'median':
            filter_function = filter_median_benito
        data_dict = benito_data
    elif dataset == 'kutsche':
        if summarization_technique == 'proximity':
            aggregation = 'robust'
        elif summarization_technique == 'mean':
            aggregation = 'mean'
        elif summarization_technique == 'median':
            aggregation = 'median'
        data_human, df, day_map = preprocess_pipeline(df_human, normalize=False, transformed=False, aggregation=aggregation)
        data_dict = kutsche_data
    else:
        data_dict = {'proximity': None, 'mean': None, 'median': None}

    if data_dict is None:
        debug_print(f"Error: data_dict is None for dataset {dataset}")
        return [], [], "", {'display': 'block'}, 'Data dictionary is not initialized.', None, None, None, '', 2

    debug_print(f"data_dict initialized: {data_dict}")

    cache_file = get_cache_filename(dataset, summarization_technique)
    cache_content = load_cache(cache_file)
    
    if cache_content is not None:
        data_dict[summarization_technique], data_human = cache_content
        debug_print(f"Loaded {dataset} data ({summarization_technique}) from cache")
    else:
        debug_print(f"No cache found for {dataset} data ({summarization_technique}), computing...")
        if dataset == 'benito':
            df_human = mapper_benito(
                datafile=os.path.join('Data', 'Benito', 'Benito_Human'),
                mappingfile=os.path.join('Data', 'Benito', 'gene_id_to_gene_name.txt'),
                map_speciment_to_gene_file=os.path.join('Data', 'Benito', 'map_speciment_to_gene.csv')
            )
            data_human, df, day_map = filter_function(df_human)
            # Remove rows with all 0's
            data_human = data_human.loc[(data_human != 0).any(axis=1)]
            data_dict[summarization_technique] = perform_gc_benito(data_human)
        elif dataset == 'kutsche':
            df_human = load_and_preprocess_kutsche(os.path.join('Data', 'Kutsche', 'genes.txt'))
            data_human, df, day_map = filter_function(df_human)
            # Remove rows with all 0's
            data_human = data_human.loc[(data_human != 0).any(axis=1)]
            data_dict[summarization_technique] = perform_gc_kutsche(data_human)
        save_cache((data_dict[summarization_technique], data_human), cache_file)
        debug_print(f"Computed and saved {dataset} data ({summarization_technique}) to cache")

    if dataset == 'intersection':
        for ds in ['kutsche', 'benito']:
            if ds == 'kutsche':
                data_dict = kutsche_data
                cache_file = get_cache_filename('kutsche', summarization_technique)
                cache_content = load_cache(cache_file)
                if cache_content is not None:
                    data_dict[summarization_technique], data_human = cache_content
                else:
                    df_human = load_and_preprocess_kutsche(os.path.join('Data', 'Kutsche', 'genes.txt'))
                    if summarization_technique == 'proximity':
                        aggregation = 'robust'
                    elif summarization_technique == 'mean':
                        aggregation = 'mean'
                    elif summarization_technique == 'median':
                        aggregation = 'median'
                    data_human, df, day_map = preprocess_pipeline(df_human, normalize=False, transformed=False, aggregation=aggregation)
                    # Remove rows with all 0's
                    data_human = data_human.loc[(data_human != 0).any(axis=1)]
                    data_dict[summarization_technique] = perform_gc_kutsche(data_human)
                    save_cache((data_dict[summarization_technique], data_human), cache_file)
            elif ds == 'benito':
                data_dict = benito_data
                cache_file = get_cache_filename('benito', summarization_technique)
                cache_content = load_cache(cache_file)
                if cache_content is not None:
                    data_dict[summarization_technique], data_human = cache_content
                else:
                    df_human = mapper_benito(
                        datafile=os.path.join('Data', 'Benito', 'Benito_Human'),
                        mappingfile=os.path.join('Data', 'Benito', 'gene_id_to_gene_name.txt'),
                        map_speciment_to_gene_file=os.path.join('Data', 'Benito', 'map_speciment_to_gene.csv')
                    )
                    if summarization_technique == 'proximity':
                        filter_function = filter_proximity_benito
                    elif summarization_technique == 'mean':
                        filter_function = filter_mean_benito
                    elif summarization_technique == 'median':
                        filter_function = filter_median_benito
                    # Remove rows with all 0's
                    data_human = data_human.loc[(data_human != 0).any(axis=1)]
                    data_human, df, day_map = filter_function(df_human)
                    data_dict[summarization_technique] = perform_gc_benito(data_human)
                    save_cache((data_dict[summarization_technique], data_human), cache_file)

        kutsche_edges = collect_significant_edges_kutsche(kutsche_data[summarization_technique], p_value_threshold=pvalue_global)
        benito_edges = collect_significant_edges_benito(benito_data[summarization_technique], p_value_threshold=pvalue_global)

        kutsche_edges = [(edge[0][0], edge[1][0], edge[0][1], edge[2]) for edge in kutsche_edges]
        benito_edges = [(edge[0][0], edge[1][0], edge[0][1], edge[2]) for edge in benito_edges]

        kutsche_df = pd.DataFrame(kutsche_edges, columns=['Gene1', 'Gene2', 'Lag', 'P_Value'])
        benito_df = pd.DataFrame(benito_edges, columns=['Gene1', 'Gene2', 'Lag', 'P_Value'])

        tf_genes_proximity = compare_datasets(kutsche_df, benito_df)
    
        
    else:
        tf_genes_proximity = data_dict[summarization_technique]

    if tf_genes_proximity is None:
        return [], [], "", {'display': 'none'}, 'No significant edges found.', None, None, None, '', 2

    #debug_print(f"Graph update: Dataset: {dataset}, Summarization Technique: {summarization_technique}, P-threshold: 0.05, Search: , Layout: dot")
    if dataset == 'intersection':
        significant_edges = tf_genes_proximity
    elif dataset == 'benito':
        debug_print(f"Filtered pairs: {benito_data[summarization_technique]}")
        significant_edges = collect_significant_edges_benito(benito_data[summarization_technique], p_value_threshold=pvalue_global)
    elif dataset == 'kutsche':
        significant_edges = collect_significant_edges_kutsche(kutsche_data[summarization_technique], p_value_threshold=pvalue_global)
    elif dataset == 'large_kutsche':
        not_needed = None # Placeholder, as we read this from the file
        filtered_pairs = filter_gene_pairs_kutsche(filepath = "granger_causality_results.csv", p_threshold=pvalue_global, starting_genes=genelist_global, higher_threshold_for_starting_genes=pvalue_global)
        significant_edges = collect_significant_edges_kutsche(filtered_pairs, p_value_threshold=pvalue_global, file=True, filepath = filtered_pairs, starting_genes=genelist_global, higher_threshold_for_starting_genes=pvalue_global)
    elif dataset == 'large_kutsche_top5%':
        filtered_pairs = filter_gene_pairs_kutsche(filepath="granger_causality_results_truncated_top5%_00004.csv",
                                                   p_threshold=pvalue_global,
                                                   starting_genes=genelist_global,
                                                   higher_threshold_for_starting_genes=pvalue_global)
        significant_edges = collect_significant_edges_kutsche(filtered_pairs,
                                                              p_value_threshold=pvalue_global,
                                                              file=True,
                                                              filepath=filtered_pairs,
                                                              starting_genes=genelist_global,
                                                              higher_threshold_for_starting_genes = pvalue_global)
    
    elif dataset == 'large_benito_human':
        not_needed = None # Placeholder, as we read this from the file
        filtered_pairs = filter_gene_pairs_benito(filepath = "granger_causality_results_benito_Human.csv", p_threshold=pvalue_global, starting_genes=genelist_global, higher_threshold_for_starting_genes=pvalue_global)
        significant_edges = collect_significant_edges_benito(filtered_pairs, p_value_threshold=pvalue_global, file=True, filepath = filtered_pairs, starting_genes=genelist_global, higher_threshold_for_starting_genes=pvalue_global)
    elif dataset == 'large_benito_gorilla':
        not_needed = None # Placeholder, as we read this from the file
        filtered_pairs = filter_gene_pairs_benito(filepath = "granger_causality_results_benito_Gorilla.csv", p_threshold=pvalue_global, starting_genes=genelist_global, higher_threshold_for_starting_genes=pvalue_global)
        significant_edges = collect_significant_edges_benito(filtered_pairs, p_value_threshold=pvalue_global, file=True, filepath = filtered_pairs, starting_genes=genelist_global, higher_threshold_for_starting_genes=pvalue_global)

    else:
        significant_edges = [] # or any default value you prefer

    debug_print(f"Computed intersection data for graph update, edges count: {len(significant_edges)}")

    G = create_network(significant_edges)
    louvain_partition = community_louvain.best_partition(G.to_undirected(), random_state=42)
    num_communities = len(set(louvain_partition.values()))

    partition = cached_apply_community_detection(pickle.dumps(G), community_detection_method, num_communities if community_detection_method != 'louvain' else None)
    debug_print("Community detection successful.")
    partition_tuple = partition_to_tuple(partition)
    community_colors = cached_assign_colors(partition_tuple)
    community_options = [{'label': html.Span(f'Community {comm}', style={'color': community_colors[comm]}), 'value': comm} for comm in set(partition.values())]

    # Ensure all communities are selected by default
    community_values = [comm['value'] for comm in community_options]  # Select all communities
    debug_print("Community detection and color assignment complete.")

    # Return the data to be stored in dcc.Store and clear search bar
    return community_options, community_values, "", {'display': 'none'}, 'Please select options and press "Send" to generate the graph.', data_human.to_dict(), dataset, summarization_technique, '', num_communities

@app.callback(
    Output('heavy-network-store', 'data'),
    [Input('apply-button', 'n_clicks')],
    [State('p-threshold-input', 'value'),
     State('dataset-dropdown', 'value'),
     State('summarization-technique-dropdown', 'value'),
     State('community-detection-method-dropdown', 'value'),
     State('num-of-communities', 'value'),
     State('num-of-consensus-runs', 'value')],
    prevent_initial_call=True
)
def compute_heavy_network(apply_click,
                          p_threshold, dataset, summarization_technique,
                          community_detection_method, num_communities, num_consensus_runs):
    global benito_data, kutsche_data
    if not dataset or not summarization_technique:
        return None

    # ---------------------------
    # Step 1: Compute significant edges (once)
    # ---------------------------
    if dataset == 'intersection':
        kutsche_edges = (collect_significant_edges_kutsche(kutsche_data[summarization_technique], 
                                                            p_value_threshold=p_threshold)
                          if kutsche_data[summarization_technique] else [])
        benito_edges = (collect_significant_edges_benito(benito_data[summarization_technique], 
                                                         p_value_threshold=p_threshold)
                        if benito_data[summarization_technique] else [])
        kutsche_edges = [(edge[0][0], edge[1][0], edge[0][1], edge[2]) for edge in kutsche_edges]
        benito_edges = [(edge[0][0], edge[1][0], edge[0][1], edge[2]) for edge in benito_edges]
        kutsche_df = pd.DataFrame(kutsche_edges, columns=['Gene1', 'Gene2', 'Lag', 'P_Value'])
        benito_df = pd.DataFrame(benito_edges, columns=['Gene1', 'Gene2', 'Lag', 'P_Value'])
        significant_edges = compare_datasets(kutsche_df, benito_df)
    elif dataset == 'benito':
        significant_edges = collect_significant_edges_benito(benito_data[summarization_technique], 
                                                              p_value_threshold=p_threshold)
    elif dataset == 'kutsche':
        significant_edges = collect_significant_edges_kutsche(kutsche_data[summarization_technique], 
                                                               p_value_threshold=p_threshold)
    elif dataset == 'large_kutsche':
        filtered_pairs = filter_gene_pairs_kutsche(filepath="granger_causality_results.csv",
                                                   p_threshold=pvalue_global,
                                                   starting_genes=genelist_global,
                                                   higher_threshold_for_starting_genes=pvalue_global)
        significant_edges = collect_significant_edges_kutsche(filtered_pairs,
                                                              p_value_threshold=pvalue_global,
                                                              file=True,
                                                              filepath=filtered_pairs,
                                                              starting_genes=genelist_global,
                                                              higher_threshold_for_starting_genes=p_threshold)
    elif dataset == 'large_kutsche_top5%':
        filtered_pairs = filter_gene_pairs_kutsche(filepath="granger_causality_results_top5_3611.csv",
                                                   p_threshold=pvalue_global,
                                                   starting_genes=genelist_global,
                                                   higher_threshold_for_starting_genes=pvalue_global)
        significant_edges = collect_significant_edges_kutsche(filtered_pairs,
                                                              p_value_threshold=pvalue_global,
                                                              file=True,
                                                              filepath=filtered_pairs,
                                                              starting_genes=genelist_global,
                                                              higher_threshold_for_starting_genes=p_threshold)
    elif dataset == 'large_benito_human':
        filtered_pairs = filter_gene_pairs_benito(filepath="granger_causality_results_benito_Human.csv",
                                                  p_threshold=pvalue_global,
                                                  starting_genes=genelist_global,
                                                  higher_threshold_for_starting_genes=pvalue_global)
        significant_edges = collect_significant_edges_benito(filtered_pairs,
                                                             p_value_threshold=pvalue_global,
                                                             file=True,
                                                             filepath=filtered_pairs,
                                                             starting_genes=genelist_global,
                                                             higher_threshold_for_starting_genes=p_threshold)
    elif dataset == 'large_benito_gorilla':
        filtered_pairs = filter_gene_pairs_benito(filepath="granger_causality_results_benito_Gorilla.csv",
                                                  p_threshold=pvalue_global,
                                                  starting_genes=genelist_global,
                                                  higher_threshold_for_starting_genes=pvalue_global)
        significant_edges = collect_significant_edges_benito(filtered_pairs,
                                                             p_value_threshold=pvalue_global,
                                                             file=True,
                                                             filepath=filtered_pairs,
                                                             starting_genes=genelist_global,
                                                             higher_threshold_for_starting_genes=p_threshold)
    else:
        significant_edges = []


    if not significant_edges:
        return None
    #debug_print(f"Significant edges: {len(significant_edges)}")
    significant_edges = set(significant_edges)
    #debug_print(f"Significant edges: {len(significant_edges)}")
    G = create_network(significant_edges)
    
    debug_print(f"Genes{ len(G.nodes)}")
    debug_print(f"Edges: {len(G.edges)}")

    # Save the gene names to a file
    # genes_names(G)

    # Run community detection (you may choose consensus or another method)
    if community_detection_method == 'consensus':
        consensus, coassoc, partitions, n_of_genes_in_interest_gene_community = consensus_partition(G, n_runs=num_consensus_runs, gene_of_interest="ZEB2")
        partition = consensus
        gene_of_interest = "ZEB2"  # adjust gene if needed
        if gene_of_interest in consensus:
            debug_print(f"Consensus: Gene {gene_of_interest} is in community {consensus[gene_of_interest]}")
        if n_of_genes_in_interest_gene_community != None:
            debug_print(f"There were {n_of_genes_in_interest_gene_community}/{len(partition)} unique genes in the community of across {num_consensus_runs} runs of Louvain in the same community as {gene_of_interest}")

    else:
        partition = cached_apply_community_detection(pickle.dumps(G), community_detection_method,
                                                     num_communities if community_detection_method != 'louvain' else None)

    # Pack the heavy network data (serialize the graph and partition)
    try:
        # heavy computation code
        graph_pickle_bytes = pickle.dumps(G)
        graph_pickle_str = base64.b64encode(graph_pickle_bytes).decode('utf-8')
        heavy_data = {'graph_pickle': graph_pickle_str, 'partition': partition}
        return heavy_data

    except Exception as e:
        debug_print("Error in heavy callback:", e)
        raise PreventUpdate

# Export genes to /n seperated file
def genes_names(network):
    # Get the gene names from the network
    gene_names = []
    with open('genes.txt', 'w') as f:
        for node in network.nodes():
            if isinstance(node, tuple):
                gene_names.append(node[0])
            else:
                gene_names.append(node)
        f.write('\n'.join(gene_names))


@app.callback(
    [Output('network-graph', 'srcDoc'),
     Output('selections-output', 'children'),
     Output('community-checklist', 'options', allow_duplicate=True),
     Output('community-checklist', 'value', allow_duplicate=True),
     Output('num-of-communities', 'value', allow_duplicate=True)],
    [Input('heavy-network-store', 'data'),
     Input('apply-communities-button', 'n_clicks')],
    [State('community-checklist', 'value'),
     State('layout-dropdown', 'value'),
     State('search-bar', 'value')],
     prevent_initial_call=True
)
def update_graph(heavy_data, apply_comm_clicks, selected_communities, layout, search_value):
    # If heavy data isn’t present, do nothing.
    #debug_print("Filtering callback triggered, heavy_data:", heavy_data)
    if not heavy_data or 'graph_pickle' not in heavy_data or 'partition' not in heavy_data:
        debug_print("No heavy data found.")
        raise PreventUpdate

    debug_print("Updating graph...")
    # Use all communities by default if nothing is selected.
    # (Assuming you compute available communities from the stored partition.)
    try:
        graph_pickle_bytes = base64.b64decode(heavy_data['graph_pickle'])
        G = pickle.loads(graph_pickle_bytes)
        partition = heavy_data['partition']
        available_communities = set(partition.values())
    except Exception as e:
        return f"<p>Error loading graph data: {e}</p>"
    if not selected_communities:
        selected_communities = list(available_communities)
    
    # (Optionally, update community colors if needed)
    partition_tuple = partition_to_tuple(partition)
    community_colors = cached_assign_colors(partition_tuple)

    community_options = [
        {'label': html.Span(f'{comm}', style={'color': community_colors.get(comm, ("#000000",))[0]}),
         'value': comm} for comm in set(partition.values())
    ]
    # Filter the graph based on the selected communities.
    nodes_to_keep = {node for node, comm in partition.items() if comm in selected_communities}
    G_filtered = G.subgraph(nodes_to_keep).copy()

    # Create the Graphviz DOT representation.
    dot = create_graphviz_dot(G_filtered, partition, community_colors, highlight_node=search_value, layout=layout)

    try:
        # REMOVE WARNINGS FROM GRAPHWIZ
        import contextlib
        with open(os.devnull, 'w') as fnull:
            with contextlib.redirect_stderr(fnull):
                graph_svg = dot.pipe(format='svg').decode('utf-8')

        graph_html = html_template.format(graph=graph_svg)
        return graph_html, "", community_options, selected_communities, len(available_communities)
    except Exception as e:
        return "", f"Error generating graph: {e}", community_options, selected_communities, len(available_communities)


@app.callback(
    [Output('community-checklist', 'value', allow_duplicate=True),
     Output('toggle-state', 'data')],
    [Input('toggle-all-button', 'n_clicks')],
    [State('community-checklist', 'options'),
     State('community-checklist', 'value'),
     State('toggle-state', 'data')],
    prevent_initial_call=True
)
def toggle_all_communities_callback(n_clicks, community_options, selected_communities, toggle_state):#
    if n_clicks is None:
        return selected_communities, toggle_state

    all_communities, new_toggle_state = toggle_all_communities(community_options, toggle_state)
    debug_print(f"Toggling all communities. New selection: {all_communities}, New toggle state: {new_toggle_state}")

    return all_communities, new_toggle_state

# Toggles all communities in the checklist
def toggle_all_communities(community_options, toggle_state):
    all_communities = [option['value'] for option in community_options]

    if toggle_state:
        # Select all communities
        new_selected_communities = all_communities
        new_toggle_state = False
    else:
        # Deselect all communities
        new_selected_communities = []
        new_toggle_state = True

    return new_selected_communities, new_toggle_state

@app.callback(
    Output('num-of-communities-container', 'style'),
    Input('community-detection-method-dropdown', 'value')
)
def show_hide_num_communities_input(method):
    if method in ['girvan_newman']:
        return {'display': 'block'}
    return {'display': 'none'}


def create_combined_plot(figures, gene_names, title):
    # Determine the layout for subplots
    num_plots = len(figures)
    num_cols = 3  # Adjust as needed
    num_rows = -(-num_plots // num_cols)  # Ceiling division

    # Create a subplot figure
    subplot_titles = []
    for gene_name in gene_names:
        subplot_titles.append(gene_name)

    subplot_fig = make_subplots(rows=num_rows, cols=num_cols, subplot_titles=subplot_titles)

    # Add each figure to the subplot
    for i, fig in enumerate(figures):
        row = i // num_cols + 1
        col = i % num_cols + 1
        for trace in fig.data:
            subplot_fig.add_trace(trace, row=row, col=col)

    # Update layout for the combined figure
    subplot_fig.update_layout(height=500*num_rows, width=500*num_cols, title_text=title)
    subplot_fig.update_xaxes(title_text="Timeperiod")
    subplot_fig.update_yaxes(title_text="Expression levels")

    return subplot_fig

@app.callback(
    Output('expression-plots', 'children'),
    [Input('search-bar', 'value'),
     Input('dataset-dropdown', 'value'),
     Input('summarization-technique-dropdown', 'value'),
     Input('p-threshold-slider', 'value'),
     Input('heavy-network-store', 'data')]
)
def show_expression_plots(search_gene, dataset, summarization_technique, p_threshold, heavy_network):
    debug_print("Debug: Entered show_expression_plots")
    if not search_gene:
        debug_print("Debug: No search gene provided")
        return ""

    if not dataset or not summarization_technique:
        debug_print("Debug: Dataset or summarization technique not selected")
        return "Please select both dataset and summarization technique."

    if not heavy_network:
        debug_print("Debug: No heavy network available")
        return "Please press 'Apply' to generate the graph."

    try:
        graph_pickle_bytes = base64.b64decode(heavy_network['graph_pickle'])
        G = pickle.loads(graph_pickle_bytes)
    except Exception as e:
        debug_print(f"Error loading graph data: {e}")
        return "Error loading graph data."

    influenced_genes = [target for source, target, data in G.edges(data=True) if source == search_gene]

    if not influenced_genes:
        debug_print(f"Debug: No genes influenced by {search_gene}")
        return f"No genes influenced by {search_gene} found."

    figures = []
    gene_names = [search_gene] + influenced_genes

    for gene in gene_names:
        try:
            df_human = heavy_network['df_human'].loc[[gene]]
            fig = plot_gene_expression(df_human, print_data=True)
            figures.append(fig)
        except KeyError:
            debug_print(f"Debug: Gene {gene} data not found in DataFrame.")

    if not figures:
        debug_print("Debug: No figures generated")
        return "No plots to display."

    title = (f"Gene Expression Plots for {search_gene} and influenced genes, "
             f"P-value Threshold: {p_threshold}, Dataset: {dataset}, "
             f"Summarization Technique: {summarization_technique}")
    combined_plot = create_combined_plot(figures, gene_names, title)
    debug_print("Debug: Combined plot created")

    return dcc.Graph(figure=combined_plot)

def plot_gene_expression(df_human, print_data=False):
    import plotly.tools as tls
    import matplotlib.pyplot as plt
    import numpy as np
    
    """
    Plot gene expression data.
    """
    debug_print("Debug: Plotting gene expression data")
    x = df_human.columns.astype(float)  # Ensure x-values are numeric
    debug_print(f"Debug: x_values - {x}")
    num_genes = df_human.shape[0]
    num_cols = 7
    num_rows = int(np.ceil(num_genes / num_cols))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, num_rows * 2.5))
    axes = axes.flatten()

    for i, (gene, series) in enumerate(df_human.iterrows()):
        ax = axes[i]
        y = series.values

        if print_data:
            debug_print(f'Gene: {gene}')
            debug_print(f'X values: {x}')
            debug_print(f'Y values: {y}')
            debug_print('---')

        ax.plot(x, y, color='dodgerblue', lw=2)
        ax.scatter(x, y, color='darkorange', alpha=0.6)
        ax.set_ylim([min(y) * 0.9, max(y) * 1.1])
        ax.set_xticks(x)  # Use the numeric x-values
        ax.set_xticklabels(x.astype(int))  # Ensure labels are integers
        ax.set_title(gene, fontsize=10, fontweight='bold')
        ax.grid(True)

    fig.text(0.5, 0.02, 'Time (days)', ha='center', va='center')
    fig.text(0.02, 0.5, 'Expression', ha='center', va='center', rotation='vertical')

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(h_pad=2, rect=[0.03, 0.03, 1, 0.95])
    debug_print("Debug: Completed plot_gene_expression")
    
    plotly_fig = tls.mpl_to_plotly(fig)
    graph = dcc.Graph(figure=plotly_fig)
    
    return graph, plotly_fig  # Return both the plotly graph and the plotly figure

def fig_to_plotly(fig):
    import plotly.tools as tls
    debug_print("Debug: Converting Matplotlib figure to Plotly")
    plotly_fig = tls.mpl_to_plotly(fig)
    debug_print("Debug: Conversion complete")
    return plotly_fig


@app.callback(
    Output('show-plots-button', 'style'),
    [Input('search-bar', 'value'),
     Input('dataset-dropdown', 'value'),
     Input('summarization-technique-dropdown', 'value'),
     Input('p-threshold-slider', 'value'),
     Input('community-detection-method-dropdown', 'value'),
     Input('num-of-communities', 'value')],
    [State('community-checklist', 'value'),
     State('current-dataset', 'data'),
     State('current-summarization-technique', 'data')]
)
def update_show_plots_button(search_value, dataset, summarization_technique, p_threshold, community_detection_method, num_communities, selected_communities, current_dataset, current_summarization_technique):
    if not search_value or dataset != current_dataset or summarization_technique != current_summarization_technique:
        return {'display': 'none'}

    global benito_data, kutsche_data

    if dataset == 'intersection':
        return {'display': 'none'}

    data_dict = benito_data if dataset == 'benito' else kutsche_data
    data = data_dict.get(summarization_technique)

    if data is None:
        return {'display': 'none'}

    # Retrieve the significant edges
    if dataset == 'benito':
        significant_edges = collect_significant_edges_benito(benito_data[summarization_technique], p_value_threshold=p_threshold)
    elif dataset == 'kutsche':
        significant_edges = collect_significant_edges_kutsche(kutsche_data[summarization_technique], p_value_threshold=p_threshold)
    elif dataset == 'large_kutsche':
        not_needed = None # Placeholder, as we read this from the file
        filtered_pairs = filter_gene_pairs_kutsche(filepath = "granger_causality_results.csv", p_threshold=p_threshold, starting_genes=genelist_global, higher_threshold_for_starting_genes=pvalue_global)
        significant_edges = collect_significant_edges_kutsche(filtered_pairs, p_value_threshold=p_threshold, file=True, filepath = filtered_pairs, starting_genes=genelist_global, higher_threshold_for_starting_genes=pvalue_global)
    elif dataset == 'large_kutsche_top5%':
        filtered_pairs = filter_gene_pairs_kutsche(filepath="granger_causality_results_top5%.csv",
                                                   p_threshold=pvalue_global,
                                                   starting_genes=genelist_global,
                                                   higher_threshold_for_starting_genes=pvalue_global)
        significant_edges = collect_significant_edges_kutsche(filtered_pairs,
                                                              p_value_threshold=pvalue_global,
                                                              file=True,
                                                              filepath=filtered_pairs,
                                                              starting_genes=genelist_global,
                                                              higher_threshold_for_starting_genes=p_threshold)
    elif dataset == 'large_benito_human':
        not_needed = None # Placeholder, as we read this from the file
        filtered_pairs = filter_gene_pairs_benito(filepath = "granger_causality_results_benito_Human.csv", p_threshold=p_threshold, starting_genes=genelist_global, higher_threshold_for_starting_genes=pvalue_global)
        significant_edges = collect_significant_edges_benito(filtered_pairs, p_value_threshold=p_threshold, file=True, filepath = filtered_pairs, starting_genes=genelist_global, higher_threshold_for_starting_genes=pvalue_global)
    elif dataset == 'large_benito_gorilla':
        not_needed = None # Placeholder, as we read this from the file
        filtered_pairs = filter_gene_pairs_benito(filepath = "granger_causality_results_benito_Gorilla.csv", p_threshold=p_threshold, starting_genes=genelist_global, higher_threshold_for_starting_genes=pvalue_global)
        significant_edges = collect_significant_edges_benito(filtered_pairs, p_value_threshold=p_threshold, file=True, filepath = filtered_pairs, starting_genes=genelist_global, higher_threshold_for_starting_genes=pvalue_global)

    else:
        significant_edges = []  # or any default value you prefer
    # Create the graph and apply community detection
    G = create_network(significant_edges)
    graph_pickle = pickle.dumps(G)
    partition = cached_apply_community_detection(graph_pickle, method=community_detection_method, num_communities=num_communities if community_detection_method != 'louvain' else None)

    # If no communities are selected, use all communities
    if not selected_communities:
        selected_communities = set(partition.values())
    nodes_to_keep = {node for node, community in partition.items() if community in selected_communities}
    G = G.subgraph(nodes_to_keep).copy()

    # Check if the search value is a node in the graph
    if search_value in G.nodes():
        return {'display': 'inline-block'}
    else:
        return {'display': 'none'}
    
@app.callback(
    Output('download-graph-html', 'data'),
    [Input('export-html-button', 'n_clicks')],
    [State('network-graph', 'srcDoc')]
)
def export_graph_as_html(n_clicks, srcdoc):
    if n_clicks and srcdoc:
        # Return the srcDoc as an HTML file for download
        return dcc.send_string(srcdoc, 'network.html')
    return None

# Callback to sync slider and input values when "Apply" button is clicked
@app.callback(
    [Output('p-threshold-slider', 'value'),
     Output('p-threshold-input', 'value')],
    [Input('apply-button', 'n_clicks')],
    [State('p-threshold-input', 'value')]
)
def sync_p_value(n_clicks, input_value):
    if n_clicks is None:
        raise PreventUpdate

    # Ensure input value stays within the slider range
    adjusted_value = min(max(input_value, 0.000001), 1)
    
    return adjusted_value, adjusted_value


# Run the app
if __name__ == '__main__':
    # Change working directory to the directory of the script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)
    
    # Initialize data dictionaries for cache
    benito_data = {'proximity': None, 'mean': None, 'median': None}
    kutsche_data = {'proximity': None, 'mean': None, 'median': None}

    # Clear cache and plots directories
    clear_directory(cache_dir)
    clear_directory(plots_dir)

    app.run_server(debug=True)
