"""Print simple metadata for significant-edge gene networks."""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sklearn.cluster import AgglomerativeClustering
from community import community_louvain
import concurrent.futures
import networkx as nx
import numpy as np
import os

from gene_analysis_common.network import create_network
from gene_analysis_benito.granger_causality import filter_gene_pairs as filter_gene_pairs_benito
from gene_analysis_benito.granger_causality import collect_significant_edges as collect_significant_edges_benito


from gene_analysis_kutsche.granger_causality import filter_gene_pairs as filter_gene_pairs_kutsche
from gene_analysis_kutsche.granger_causality import collect_significant_edges as collect_significant_edges_kutsche



if __name__ == '__main__':
    # Change working directory to the directory of the script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    # Parameters
    #filepaths = {"granger_causality_results_explore_top5_00005.csv": 0.0005,
     #            "granger_causality_results_explore_top5_0001.csv": 0.001,
      #              "granger_causality_results_explore_top5_0005.csv": 0.005,
       #             "granger_causality_results_truncated.csv": 0.001,
        #        }
        #
    filepaths = {
                 "granger_causality_results_truncated.csv": 0.0015,
                 #"granger_causality_results_truncated.csv": 0.002,
                 }
    genelist_global = ["ZEB2"]
    notable_genes = ["ZEB2", "MECP2", "FMR1"]

    # Load gene pairs and build the network
    for data in filepaths.items():
        filtered_pairs = filter_gene_pairs_kutsche(filepath=data[0],
                                                p_threshold=data[1],
                                                starting_genes=genelist_global,
                                                higher_threshold_for_starting_genes=data[1])

        significant_edges = collect_significant_edges_kutsche(filtered_pairs,
                                                            p_value_threshold=data[1],
                                                            file=True,
                                                            filepath=filtered_pairs,
                                                            starting_genes=genelist_global,
                                                            higher_threshold_for_starting_genes=data[1])
        G = create_network(significant_edges)
        H = G.to_undirected() # removes directionality for analysis, also merges multi-edges
        print(f"\nAnalysing: {data[0]}")
        print(f"Threshold for p-value: {data[1]}")
        print(f"Number of edges in the network: {len(G.edges)}")
        print(f"Number of nodes in the network: {len(G.nodes)}")
        print(f"Number of genes in Top 5% + ZEB2: {math.floor((len(G.nodes)+1)*0.05)}\n")  # +1 for ZEB2
        for notable_gene in notable_genes:
            if notable_gene in G.nodes:
                print(f"{notable_gene} is present in the network.")
                print(f"Number of neighbors for {notable_gene}: {len(list(H.neighbors(notable_gene)))}")
                print(f"Neighbors for {notable_gene}: {list(H.neighbors(notable_gene))}\n")
            else:
                print(f"{notable_gene} is NOT present in the network.")
