#!/bin/bash

# The file to extract genes from
data_file="Kutsche_Counts.txt"

# The output file containing unique genes
output_file="kutsche_all_genes.txt"

# Check if data file exists
if [ ! -f "$data_file" ]; then
    echo "Error: Data file '$data_file' does not exist."
    exit 1
fi

# Extract header
head -n 1 "$data_file" > "$output_file"

# Extract genes (assuming gene names are in the first column)
# Skip header, extract only first column, remove duplicates
tail -n +2 "$data_file" | awk '{print $1}' | sort | uniq >> "$output_file"

# Report number of unique genes extracted
gene_count=$(($(wc -l < "$output_file") - 1))
echo "Unique genes from '$data_file' have been copied to '$output_file'."
echo "Total unique genes copied: $gene_count"
