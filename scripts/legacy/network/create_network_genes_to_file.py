"""Build a significant-edge network and report/export its gene members."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gene_analysis_common.network import create_network
from gene_analysis_common.granger_causality import filter_gene_pairs as filter_gene_pairs_kutsche
from gene_analysis_common.granger_causality import collect_significant_edges as collect_significant_edges_kutsche
import os
import networkx as nx
import math


if __name__ == "__main__":
    # Change working directory to the directory of the script
    abspath = os.path.abspath(__file__)
    dname = os.path.dirname(abspath)
    os.chdir(dname)

    filepaths = {
                 "MECP2_1_5_probing_explore_granger_results_p005_479853of13546008.csv": 0.001,
                 #"granger_causality_results_truncated.csv": 0.002,
                 }
    genelist_global = ["MECP2"]

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


#        with open(f"network_genes_{data[1]:.4f}_MECP2.txt", "w") as f:
 #           for node in G.nodes():
 #               f.write(f"{node}\n")
        print(f"Saved gene list to network_genes_{data[1]:.4f}_MECP2.txt")
        print(f"Total genes in network for p_threshold {data[1]:.4f}: {G.number_of_nodes()}")
