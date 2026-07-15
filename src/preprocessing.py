

import os
import hashlib
import pandas as pd
from sklearn.model_selection import train_test_split
from data_loader import build_dataframe


def file_hash(filepath: str) -> str:
    """Compute an MD5 hash of a file's contents."""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate images based on file content (not filename).
    Keeps the first occurrence of each unique image, drops the rest.
    """
    df = df.copy()
    print("Hashing all images to find duplicates (this may take a minute)...")
    df["hash"] = df["filepath"].apply(file_hash)

    before = len(df)
    df = df.drop_duplicates(subset="hash", keep="first")
    after = len(df)

    print(f"Removed {before - after} duplicate images ({before} -> {after})")
    return df.drop(columns="hash").reset_index(drop=True)


def create_splits(train_dir: str, val_size: float = 0.15, random_state: int = 42):
    df = build_dataframe(train_dir)
    df = deduplicate(df)  

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df["label"],
        random_state=random_state,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


if __name__ == "__main__":
    train_df, val_df = create_splits("data/raw/Training")

    print(f"\nTrain: {len(train_df)} images")
    print(f"Validation: {len(val_df)} images")

    print("\nTrain class distribution:")
    print(train_df["label"].value_counts())

    print("\nValidation class distribution:")
    print(val_df["label"].value_counts())

    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train_split.csv", index=False)
    val_df.to_csv("data/processed/val_split.csv", index=False)
    print("\nSaved splits to data/processed/")