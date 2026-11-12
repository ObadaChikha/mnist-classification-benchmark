# AI Methods for the MNIST Dataset

## Overview
A systematic machine learning benchmark evaluating classical supervised algorithms and advanced ensemble architectures for multi-class handwritten digit classification on the MNIST dataset (70,000 flattened 28×28 images; 60,000 train / 10,000 test split).

📄 **Full Project Documentation & Logbook:** See [`MNIST_Classification_Report.pdf`](./MNIST_Classification_Report.pdf) for detailed experiment logs, scaling diagnostics, and error analysis.

## Key Engineering & Analytical Highlights
* **Data Scaling Diagnostics:** Demonstrated experimentally that `StandardScaler` degrades distance-based KNN performance on sparse image data (dropping accuracy from `97.17%` to `94.69%`) by shifting true zero-background pixels into artificial negative values, whereas unscaled or `[0, 1]` normalized pixels preserve spatial contrast.
* **Leakage Prevention & Error Analysis:** Identified and resolved early evaluation leakage, established strict `StratifiedKFold` validation, and analyzed pixel-level misclassification patterns across ambiguous handwritten digits.
* **Ensemble Architecture Exploration:** Investigated why naive Hard Voting underperformed individual strong learners ("democracy trap") and resolved it via probability-weighted **Soft Voting**, **HistGradientBoosting**, and **Stacking**.

## Model Progression & Benchmark Results (Test Accuracy)

| Model / Architecture | Test Accuracy | Key Findings & Configuration |
| :--- | :---: | :--- |
| **Decision Tree** (Baseline) | `88.10%` | Tuned to `max_depth=15`; prone to rigid pixel-level overfitting |
| **AdaBoost** | `91.83%` | Over-sensitive to noisy/distorted handwritten outliers |
| **Logistic Regression** | `92.44%` | Tuned (`C=100.0`, `max_iter=1000`); highlights linear decision boundaries on image data |
| **K-Nearest Neighbors (KNN)** | `97.17%` | Optimal at `k=3` with `weights='distance'` on unscaled pixels |
| **Random Forest** | `97.25%` | `1,000` trees (`max_depth=None`); effectively eliminates single-tree variance |
| **HistGradientBoosting** | **`97.84%`** | Learns efficiently from residuals; robust against image noise |
| **Stacking Classifier** | **`97.84%`** | **Level-0 (`KNN` + `Random Forest`) + Level-1 (`Logistic Regression`) meta-learner** |

## Tech Stack
* **Python:** `Scikit-Learn`, `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`
