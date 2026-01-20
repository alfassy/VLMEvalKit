import pandas as pd
import os

def create_small():
    print("Reading ChartCap.tsv...")
    # Use chunksize or nrows to avoid memory issues if file is huge, though we just want 12 rows.
    # We must assume 'ChartCap.tsv' is in CWD as per previous steps.
    try:
        df = pd.read_csv('ChartCap.tsv', sep='\t', nrows=12)
        print(f"Read {len(df)} rows.")
        output = 'ChartCapSmall.tsv'
        df.to_csv(output, sep='\t', index=False)
        print(f"Saved {output} with {len(df)} samples.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    create_small()
