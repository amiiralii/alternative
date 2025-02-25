import os
import pandas as pd
import glob

# Define treatments and ranks
treatments = ["lgbm", "lr", "ours", "baseline"]
ranks = [0, 1, 2, 3]

# Initialize a dictionary to store counts
counts = {treatment: {rank: 0 for rank in ranks} for treatment in treatments}

# Get all CSV files in the current directory
csv_files = glob.glob("results/parsed/*.csv")
# Process each CSV file
for file in csv_files:
    with open(file, "r") as f:
        for line in f:
            # Skip lines that don't start with a rank (0-3)
            parts = [p.strip() for p in line.split(",")]
            if not parts[0].isdigit():
                continue
            
            # Split the line by comma and clean up
            if len(parts) < 2:
                continue
            try:
                rank = int(parts[0])
                treatment = parts[1].lower()
                # Update counts if valid treatment and rank
                if treatment in counts and rank in ranks:
                    counts[treatment][rank] += 1
            except ValueError:
                continue  # Skip malformed lines

# Convert to DataFrame for better visualization
df = pd.DataFrame.from_dict(counts, orient="index", columns=ranks)
df = df.set_axis(["Rank 0", "Rank 1", "Rank 2", "Rank 3"], axis=1)
df = df.sort_values(by=["Rank 0"],ascending=False)
df.to_csv("performance.csv", sep=',', encoding='utf-8')