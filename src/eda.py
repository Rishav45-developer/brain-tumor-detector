import sys
import os
import random

import matplotlib.pyplot as plt
import cv2

from data_loader import build_dataframe, class_distribution

# 1. Load data
train_df = build_dataframe('data/raw/Training')
print(train_df.head())

# 2. Class distribution
counts = class_distribution(train_df)
print("\nClass distribution (Training):")
print(counts)

counts.plot(kind='bar', title='Training set class distribution')
plt.ylabel('Number of images')
plt.savefig('reports/class_distribution.png')
print("\nSaved chart to reports/class_distribution.png")
plt.show()

# 3. Sample images per class
classes = train_df['label'].unique()
fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 4))

for ax, cls in zip(axes, classes):
    sample_path = train_df[train_df['label'] == cls]['filepath'].iloc[0]
    img = cv2.imread(sample_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    ax.set_title(cls)
    ax.axis('off')

plt.tight_layout()
plt.savefig('reports/sample_images.png')
print("Saved sample images to reports/sample_images.png")
plt.show()

# 4. Check image sizes/formats
print("\nSample image shapes:")
sample_paths = random.sample(list(train_df['filepath']), 10)
for p in sample_paths:
    img = cv2.imread(p)
    print(os.path.basename(p), '-> shape:', img.shape)

