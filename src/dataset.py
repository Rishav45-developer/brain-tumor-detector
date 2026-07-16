

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


LABEL_MAP = {"glioma": 0, "meningioma": 1, "notumor": 2, "pituitary": 3}


train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class BrainMRIDataset(Dataset):
    def __init__(self, csv_path: str, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["filepath"]).convert("RGB")
        label = LABEL_MAP[row["label"]]

        if self.transform:
            image = self.transform(image)

        return image, label


if __name__ == "__main__":
    train_ds = BrainMRIDataset("data/processed/train_split.csv", transform=train_transform)
    val_ds = BrainMRIDataset("data/processed/val_split.csv", transform=eval_transform)

    print(f"Train dataset size: {len(train_ds)}")
    print(f"Validation dataset size: {len(val_ds)}")

    image, label = train_ds[0]
    print(f"\nSample image tensor shape: {image.shape}")
    print(f"Sample label: {label}")