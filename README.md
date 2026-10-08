# PCA-Based Face Recognition

## Overview

This project demonstrates the practical application of **Linear Algebra and PCA (Principal Component Analysis)** in **face recognition**.

Face images are converted into numerical vectors, reduced to important features using PCA, and classified using **K-Nearest Neighbors (KNN)**.

## Technologies Used

- Python
- NumPy
- Matplotlib
- Scikit-learn
- Pillow

## Dataset

The project can use:
- A custom dataset stored in `pca_face_dataset/`
- The **LFW (Labeled Faces in the Wild)** dataset

Images are converted to grayscale and resized to **64 × 64 pixels**.

## Project Files

```text
PCA-Face-Recognition/
│
├── pca.py
├── create_dataset.py
├── pca_face_dataset/
└── results/
```

## How to Run

### 1. Install dependencies

```bash
pip install numpy matplotlib scikit-learn pillow
```

### 2. Create the dataset

```bash
python create_dataset.py
```

This downloads LFW faces and organizes them into the `pca_face_dataset` folder.

### 3. Run the project

```bash
python pca.py
```

## What the Program Does

1. Loads and preprocesses face images.
2. Splits the dataset into training and testing data.
3. Applies PCA to extract important facial features.
4. Generates **mean face and eigenface** visualizations.
5. Projects faces into 2D PCA space.
6. Tests different numbers of PCA components.
7. Uses KNN for face recognition.
8. Calculates recognition accuracy.
9. Saves graphs and CSV results.

## Output

The `results/` folder contains:

- `eigenfaces.png`
- `pca_2d_projection.png`
- `accuracy_vs_components.png`
- `predictions.png`
- `accuracy_vs_components.csv`
- `predictions.csv`

## Main Concepts

**Linear Algebra → PCA → Eigenfaces → Dimensionality Reduction → KNN → Face Recognition**

## Conclusion

This project shows how mathematical concepts such as **vectors, projections, eigenvectors, PCA, and variance** can be applied to a practical **AI face recognition system**.