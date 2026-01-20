import os
from vlmeval.dataset import ChartCapDataset

def verify_probe():
    # Set LMUData to current directory where ChartCapProbe.tsv was created
    os.environ['LMUData'] = os.getcwd()
    
    print("Instantiating ChartCapProbe...")
    ds = ChartCapDataset(dataset='ChartCapProbe')
    print(f"Dataset length: {len(ds)}")
    if len(ds) == 5000:
        print("SUCCESS: Dataset length is correct.")
    else:
        print(f"FAILURE: Expected 5000 samples, got {len(ds)}.")

if __name__ == "__main__":
    verify_probe()
