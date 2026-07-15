
import hashlib
import pandas as pd


def file_hash(filepath: str) -> str:
    """Compute an MD5 hash of a file's contents."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def find_duplicates(df: pd.DataFrame) -> pd.DataFrame:
   
    df = df.copy()
    df["hash"] = df["filepath"].apply(file_hash)

    duplicate_hashes = df["hash"][df["hash"].duplicated(keep=False)]
    duplicates = df[df["hash"].isin(duplicate_hashes)]

    return duplicates.sort_values("hash")


if __name__ == "__main__":
    train_df = pd.read_csv("data/processed/train_split.csv")
    val_df = pd.read_csv("data/processed/val_split.csv")

    print("Checking training set for internal duplicates...")
    train_dupes = find_duplicates(train_df)
    print(f"Found {len(train_dupes)} duplicate rows within training set")

    print("\nChecking validation set for internal duplicates...")
    val_dupes = find_duplicates(val_df)
    print(f"Found {len(val_dupes)} duplicate rows within validation set")

    
    print("\nChecking for duplicates leaking between train and validation...")
    train_df["hash"] = train_df["filepath"].apply(file_hash)
    val_df["hash"] = val_df["filepath"].apply(file_hash)

    leaked_hashes = set(train_df["hash"]) & set(val_df["hash"])
    print(f"Found {len(leaked_hashes)} hashes present in BOTH train and validation")

    if leaked_hashes:
        print("\nWARNING: Data leakage detected. Example leaked file pairs:")
        for h in list(leaked_hashes)[:5]:
            t = train_df[train_df["hash"] == h]["filepath"].values
            v = val_df[val_df["hash"] == h]["filepath"].values
            print(f"  Train: {t[0]}  <-->  Val: {v[0]}")
    else:
        print("No leakage between train and validation splits. Good.")