

import pandas as pd
from data_loader import build_dataframe
from check_duplicates import file_hash

if __name__ == "__main__":
    train_df = pd.read_csv("data/processed/train_split.csv")
    val_df = pd.read_csv("data/processed/val_split.csv")
    test_df = build_dataframe("data/raw/Testing")

    print("Hashing train, val, and test sets (may take a minute)...")
    train_df["hash"] = train_df["filepath"].apply(file_hash)
    val_df["hash"] = val_df["filepath"].apply(file_hash)
    test_df["hash"] = test_df["filepath"].apply(file_hash)

    trainval_hashes = set(train_df["hash"]) | set(val_df["hash"])
    leaked = trainval_hashes & set(test_df["hash"])

    print(f"\nFound {len(leaked)} test images duplicated in train/val")
    if leaked:
        print("WARNING: leakage detected. Examples:")
        for h in list(leaked)[:5]:
            match = test_df[test_df["hash"] == h]["filepath"].values[0]
            print(" Leaked:", match)
    else:
        print("Test set is clean and fully independent.")