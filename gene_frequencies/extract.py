import pandas as pd

# Parameters (set these to your desired values)
csv_file = 'ZEB2_coassoc_12000_runs.csv'
output_file = 'top_genes.txt'
x = 93  # Replace with the number of genes you want to extract
column_name = 'Gene'  # Replace with the correct column name if needed

# Read the CSV file
df = pd.read_csv(csv_file)

# Extract the top x genes
top_genes = df[column_name].head(x)

# Write the genes to a text file
with open(output_file, 'w') as file:
    for gene in top_genes:
        file.write(f"{gene}\n")

print(f"Top {x} genes have been written to {output_file}")