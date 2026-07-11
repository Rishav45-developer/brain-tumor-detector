import os
import pandas as pd

def build_dataframe(root_dir: str) -> pd.DataFrame:
    records = []

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"'{root_dir}' does not exist.")

    class_names = sorted(os.listdir(root_dir))

    for class_name in class_names:
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        for filename in os.listdir(class_dir):
            if filename.lower().endswith((".jpg", ".jpeg", ".png")):
                filepath = os.path.join(class_dir, filename)
                records.append({"filepath": filepath, "label": class_name})

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError(f"No images found in '{root_dir}'.")

    return df


def class_distribution(df: pd.DataFrame) -> pd.Series:
    return df["label"].value_counts()


if __name__ == "__main__":
    train_df = build_dataframe("data/raw/Training")
    test_df = build_dataframe("data/raw/Testing")

    print(f"Training images: {len(train_df)}")
    print(f"Testing images: {len(test_df)}")
    print("\nClass distribution (Training):")
    print(class_distribution(train_df))