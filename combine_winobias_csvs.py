"""
Combine all 4 WinoBias fine-tuning CSV files into a single CSV.
"""

import pandas as pd
import os

# Input directory and files
input_dir = "data/winobias_split"
csv_files = [
    "type1_pro_finetune.csv",
    "type1_anti_finetune.csv",
    "type2_pro_finetune.csv",
    "type2_anti_finetune.csv"
]

# Read and combine all CSV files
dfs = []
for csv_file in csv_files:
    file_path = os.path.join(input_dir, csv_file)
    df = pd.read_csv(file_path)
    print(f"Loaded {csv_file}: {len(df)} samples")
    dfs.append(df)

# Concatenate all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# Save combined CSV
output_path = os.path.join(input_dir, "winobias_finetune_validation.csv")
combined_df.to_csv(output_path, index=False)

print(f"\n✅ Combined {len(combined_df)} total samples")
print(f"✅ Saved to: {output_path}")
print(f"\nPreview:")
print(combined_df.head(3))
