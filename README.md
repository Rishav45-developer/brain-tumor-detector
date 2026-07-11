# Brain Tumor Classification from MRI Scans

A deep learning research project for classifying brain MRI scans into four
categories — glioma, meningioma, pituitary tumor, and no tumor — using
convolutional neural networks and transfer learning. The project follows a
structured experimental methodology, including a literature comparison,
ablation studies, and Grad-CAM based model interpretability.

**Disclaimer:** This is an educational and research project. It is not a
medical diagnostic tool, has not undergone clinical validation, and must not
be used to make real medical decisions. All data is drawn from a public
research dataset.

## Overview

Given a brain MRI scan, the model predicts whether it shows a tumor and, if
so, which type. Beyond training a classifier, the project aims to:

- Document a reproducible experimental methodology
- Compare results against existing published approaches
- Run ablation studies to evaluate the impact of individual design choices
- Provide model interpretability through Grad-CAM visualizations

## Dataset

- Source: Brain Tumor MRI Dataset (Kaggle)
- Classes: glioma, meningioma, pituitary, notumor
- Training set: 5,600 images
- Test set: 1,600 images
- Class distribution: balanced, 1,400 images per class in training

Raw images are not included in this repository. See `data/raw/README.md`
for instructions on obtaining the dataset.

## Methodology

| Stage | Description |
|---|---|
| Baseline model | Convolutional neural network trained from scratch |
| Transfer learning | Fine-tuning of a pretrained CNN backbone |
| Evaluation | Accuracy, per-class precision/recall/F1, ROC-AUC, confusion matrix |
| Interpretability | Grad-CAM visualizations on correct and misclassified samples |
| Experimental rigor | Fixed random seeds, repeated runs reported as mean ± standard deviation, ablation studies |

Detailed reasoning behind each design decision is documented in
`reports/methodology.md`.

## Tech Stack

- Language: Python 3.11
- Deep learning: PyTorch, torchvision
- Data analysis: NumPy, Pandas, Matplotlib, Seaborn, OpenCV
- Evaluation: scikit-learn
- Interpretability: Grad-CAM
- Deployment: FastAPI / Streamlit
- Environment management: Python venv

## Project Structure

```
brain-tumor-detector/
├── README.md
├── ROADMAP.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── week1_eda.ipynb
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── explain.py
├── models/
├── app/
├── reports/
└── tests/
```

## Setup

```bash
git clone https://github.com/<your-username>/brain-tumor-detector.git
cd brain-tumor-detector

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Download the dataset manually following the instructions in
`data/raw/README.md`.

Verify the setup:

```bash
python src/data_loader.py
```

Expected output:

```
Training images: 5600
Testing images: 1600

Class distribution (Training):
glioma        1400
meningioma    1400
notumor       1400
pituitary     1400
```

## Results

To be added as training and evaluation are completed. See
`reports/final_report.md` for the full write-up.

## Reports

Progress reports, the literature review, methodology, and final report are
maintained in `reports/`.

## License and Attribution

This project uses the Brain Tumor MRI Dataset from Kaggle, compiled from
multiple public sources (Figshare, SARTAJ, Br35H). It is used here strictly
for educational and research purposes.