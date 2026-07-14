
import os
from sklearn.model_selection import train_test_split
from data_loader import build_dataframe


def create_splits(train_dir: str, val_size: float = 0.15, random_state: int = 42):
    """
    Load the training folder and split it into train/val sets.
    Stratified split keeps class balance consistent in both sets.
    """
    df = build_dataframe(train_dir)

    train_df, val_df = train_test_split(
        df,
        test_size=val_size,
        stratify=df["label"],       # preserves class balance in both splits
        random_state=random_state,  # same split every time we run this
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True)


if __name__ == "__main__":
    train_df, val_df = create_splits("data/raw/Training")

    print(f"Train: {len(train_df)} images")
    print(f"Validation: {len(val_df)} images")

    print("\nTrain class distribution:")
    print(train_df["label"].value_counts())

    print("\nValidation class distribution:")
    print(val_df["label"].value_counts())

    # Save splits to disk so every future script (training, evaluation)
    # uses the EXACT same split — prevents accidental data leakage
    # between experiments.
    os.makedirs("data/processed", exist_ok=True)
    train_df.to_csv("data/processed/train_split.csv", index=False)
    val_df.to_csv("data/processed/val_split.csv", index=False)
    print("\nSaved splits to data/processed/")