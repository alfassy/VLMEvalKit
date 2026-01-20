import os
from vlmeval.dataset import ChartCapDataset

def verify_small():
    # Set LMUData to current directory where ChartCapSmall.tsv was created
    os.environ['LMUData'] = os.getcwd()
    
    print("Instantiating ChartCapSmall...")
    ds = ChartCapDataset(dataset='ChartCapSmall')
    print(f"Dataset length: {len(ds)}")
    if len(ds) == 12:
        print("SUCCESS: Dataset length is correct.")
    else:
        print(f"FAILURE: Expected 12 samples, got {len(ds)}.")

if __name__ == "__main__":
    verify_small()
