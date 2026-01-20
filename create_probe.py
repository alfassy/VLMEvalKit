import pandas as pd
import os

def create_probe():
    print("Reading ChartCap.tsv...")
    try:
        # Read 5000 rows
        df = pd.read_csv('ChartCap.tsv', sep='\t', nrows=5000)
        print(f"Read {len(df)} rows.")
        output = 'ChartCapProbe.tsv'
        df.to_csv(output, sep='\t', index=False)
        print(f"Saved {output} with {len(df)} samples.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    create_probe()
