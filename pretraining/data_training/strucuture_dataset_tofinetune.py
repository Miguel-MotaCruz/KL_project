import csv
import pandas as pd
from typing import Tuple

def load_csv(csv_path):
    try:
        with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
            data = pd.read_csv(fh)
            print(f"Loaded {len(data)} rows from {csv_path}\n{data.head()}")
        return data
    except Exception as e:
        print(f"Error loading CSV file {csv_path}: {e}")
        return None    

def transform_to_finetuning_format(input_csv_path: str, output_csv_path: str):
    """Transform triple dataset to prompt-completion format using vectorized operations."""
    # Load data
    df = load_csv(input_csv_path)
    if df is None:
        return
    
    # Vectorized string operations - operates on entire columns at once
    df['prompt'] = df['subject'] + ', ' + df['relation'] + ', [mask]'
    df['completion'] = df['object']
    
    # Keep only the new columns
    output_df = df[['prompt', 'completion']]
    
    # Save
    output_df.to_csv(output_csv_path, index=False, encoding='utf-8')
    print(f"\n✅ Saved {len(output_df)} rows to {output_csv_path}")
    print(f"Preview:\n{output_df.head()}")
    
    return output_df

if __name__ == "__main__":
    # Example usage

    # balanced_numbers = [1,2,3,4,5,6,7,10,13,15,20,50,100]
    # for balanced_number in balanced_numbers:
    #     input_path = f"/Users/miguel_cruz/Documents/Miguel_Cruz/LEI/5ano/1semestre/KL/KL_src/pretraining/data_training/dataset_balanced{balanced_number}.csv"
    #     output_path = f"/Users/miguel_cruz/Documents/Miguel_Cruz/LEI/5ano/1semestre/KL/KL_src/pretraining/data_training/train_{balanced_number}bal.csv"
    #     transform_to_finetuning_format(input_path, output_path)
    
    imbalanced_numbers = [8, 40, 100, 200]
    for imbalanced_number in imbalanced_numbers:
        input_path = f"/Users/miguel_cruz/Documents/Miguel_Cruz/LEI/5ano/1semestre/KL/KL_src/pretraining/data_training/dataset_imbalanced{imbalanced_number}.csv"
        output_path = f"/Users/miguel_cruz/Documents/Miguel_Cruz/LEI/5ano/1semestre/KL/KL_src/pretraining/data_training/train_{imbalanced_number}imbal.csv"
        
        transform_to_finetuning_format(input_path, output_path)
