# Week 2 Report

## Goals for this week
- Build a train/validation split from the training data
- Check the dataset for duplicate images and data leakage
- Build a PyTorch Dataset and DataLoader pipeline
- Design a baseline CNN architecture
- Set up the training loop

## What I did (day by day)

**Day 1** — Wrote `src/preprocessing.py` to split the training data into
train and validation sets using a stratified split, so class balance is
preserved in both sets. Saved the split to `data/processed/train_split.csv`
and `data/processed/val_split.csv` so every future script uses the exact
same split.

**Day 2** — Wrote `src/check_duplicates.py` to check for duplicate images
based on file content (not filename). Found 42 images duplicated between
the training and validation sets — a real data leakage issue, since the
same image appearing in both sets would let the model be evaluated on
data it had effectively already seen during training.

**Day 3** — Updated `src/preprocessing.py` to deduplicate the dataset
*before* splitting, rather than after. Removed 171 duplicate images
(5,600 → 5,429) prior to the split. Re-ran the duplicate check afterward
and confirmed zero leakage between train and validation. Also wrote
`src/check_test_leakage.py` to confirm the held-out Testing set has no
overlap with the training or validation data — confirmed clean, 0 leaked
images.

**Day 4** — Wrote `src/dataset.py`: a PyTorch `Dataset` class
(`BrainMRIDataset`) that loads images from the split CSVs, resizes them
to 224×224, applies light augmentation (horizontal flip) to training
data only, and normalizes pixel values using standard ImageNet
statistics. Wrapped both datasets in `DataLoader` objects for batching.
Verified output shapes: batches of `[32, 3, 224, 224]` images with
`[32]` labels.

**Day 5** — Wrote `src/model.py`: a baseline CNN built from scratch,
with three convolutional blocks (32 → 64 → 128 channels) followed by a
fully connected classification head with dropout. Verified the
architecture with a sanity check using a random fake batch, confirming
the expected output shape of `[batch_size, 4]` (one score per class).

**Day 6** — Wrote `src/train.py`: the full training loop, including
loss calculation (cross-entropy), the Adam optimizer, per-epoch training
and validation functions, and logic to save the model checkpoint only
when validation accuracy improves. 

## Key findings

- Original dataset contained 171 duplicate images, later found to
  include 42 that had leaked across the train/validation split — a
  genuine data quality issue caught before it could silently inflate
  results.
- After deduplication, the class distribution is close to but no longer
  perfectly balanced (duplicates were not evenly distributed across
  classes).
- Final clean dataset: 4,614 training images, 815 validation images,
  1,600 test images — all confirmed free of cross-set duplication.

## Engineering Decisions

- Deduplicated the dataset by file content hash (MD5) rather than
  filename, since identical images can exist under different filenames
  across the combined data sources (Figshare, SARTAJ, Br35H).
- Saved train/validation splits to CSV rather than re-splitting randomly
  in every script, to guarantee every experiment uses the exact same
  data and stays reproducible and comparable.
- Used a fixed random seed (42) throughout for reproducibility.
- Built a simple CNN from scratch as a baseline before moving to transfer
  learning, to establish an honest performance floor to compare against.
- Applied light augmentation (horizontal flip only) to avoid distorting
  medically meaningful orientation in MRI scans.

## Plan for next week

Run the baseline CNN training, evaluate results (accuracy, per-class
precision/recall/F1, confusion matrix), then move to transfer learning
with a pretrained backbone for comparison.