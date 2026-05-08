"""Legacy helper for expanding significant edges connected to seed genes."""

import csv

def filter_edges_connected_to_starting_genes(
    filepath, p_threshold, starting_genes, output_path,
    higher_threshold_for_starting_genes=0.001
):
    """Write all significant edges connected to an expanding seed-gene set."""
    all_related_genes = set(starting_genes)
    newly_added_genes = set(starting_genes)
    filtered_edges = []

    # Read in ALL candidate edges that pass either threshold rule
    with open(filepath, 'r', newline='') as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        edges = []
        for row in reader:
            try:
                pvalue = float(row['p-value'])
            except:
                continue
            gene1 = row['gene1']
            gene2 = row['gene2']
            if (
                pvalue <= p_threshold or
                ((gene1 in starting_genes or gene2 in starting_genes) and pvalue <= higher_threshold_for_starting_genes)
            ):
                edges.append(row)

    # Iteratively expand the subnetwork, just like your function
    final_rows = []
    already_seen = set()
    while newly_added_genes:
        current_genes = newly_added_genes.copy()
        newly_added_genes.clear()
        for row in edges:
            gene1 = row['gene1']
            gene2 = row['gene2']
            pair = (gene1, gene2)
            if pair in already_seen:
                continue
            if gene1 in all_related_genes or gene2 in all_related_genes:
                final_rows.append(row)
                already_seen.add(pair)
                # Expand the set
                if gene1 not in all_related_genes:
                    newly_added_genes.add(gene1)
                if gene2 not in all_related_genes:
                    newly_added_genes.add(gene2)
                all_related_genes.update([gene1, gene2])

    # Write only filtered rows to the new CSV
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(final_rows)

    print(f"Filtered file written to {output_path}")
    print(f"Total connected edges: {len(final_rows)}")

if __name__ == "__main__":
    filter_edges_connected_to_starting_genes(
        filepath="Material/SM2/SMTable2.csv",
        p_threshold=0.0005,
        starting_genes=["ZEB2"],
        output_path="connected_edges.csv",
    )
