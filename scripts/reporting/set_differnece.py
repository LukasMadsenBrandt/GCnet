"""Compare generated gene lists against the curated seed gene set."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gene_analysis.io.paths import data_path, results_path


def load_genes(filepath):
    """Load a one-gene-per-line file into a set."""
    with open(filepath, "r") as f:
        # Strip whitespace and ignore empty lines
        return set(line.strip() for line in f if line.strip())
    

#genes1_files = ["13809_00015.csv", "16090_0002.csv"]
genes1_files = [
    data_path("Kutsche", "0002", "gene_names_0002_115.txt"),
    data_path("Kutsche", "00015", "gene_names_00015_115.txt"),
]

original_list = data_path("Kutsche", "unique_genes.txt")
gene2 = load_genes(original_list)
writeToFile = False
for file in genes1_files:
    gene1 = load_genes(file)

    missing_genes = gene1 - gene2
    known_genes = gene2 & gene1

    # Output result
    if writeToFile == True:
        info_path = results_path("gene_lists", f"{Path(file).name}_information.txt")
        with open(info_path, "w") as f:
            f.write(f"File {file}:\n\n")
            f.write(f"Number of new genes: {len(missing_genes)}\n\n")
            f.write("Gene names (sorted by frequency of being in the same community in ZEB2):\n")
            for gene in sorted(missing_genes):
                f.write(gene + "\n")
            f.write(f"\n")
            f.write(f"Number of already known genes: {len(known_genes)}\n\n")
            f.write("Gene names (sorted by frequency of being in the same community in ZEB2):\n")
            for gene in sorted(known_genes):
                f.write(gene + "\n")

    else:
        print(f"File {file}:\n")
        print(f"Number of new genes: {len(missing_genes)}\n")
        #for gene in sorted(missing_genes):
        #    print(gene)
        print(f"\n")
        print(f"Number of already known genes: {len(known_genes)}\n")
        #for gene in sorted(known_genes):
        #    print(gene)
