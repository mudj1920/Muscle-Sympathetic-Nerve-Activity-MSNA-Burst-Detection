# Muscle Sympathetic Nerve Activity (MSNA) Burst Detection

**Goal**: Detect bursts in MSNA signals using a **machine learning** approach, while satisfying these core requirements:
1. The solution **must** be machine-learning based and learn from training data.
2. The **out-of-fold F1 score** must be reported using the provided `msna_metric()` function without leaking data.
3. The solution **must** be runnable on an 8GB GPU (or CPU) in under 6 hours, including training.

This repository provides a reference pipeline that:
- Loads, filters, and normalizes MSNA data.
- Segments the data into meaningful units (ECG cycles).
- Extracts handcrafted features (e.g., amplitude, position, cycle length).
- Trains a machine learning model (Random Forest) to predict bursts within each ECG cycle.
- Evaluates model performance via **5-fold cross-validation** using the `msna_metric()` F1 score.

---

## Contents
1. [Introduction](#introduction)
2. [Data and Preprocessing](#data-and-preprocessing)
3. [Machine Learning Approach](#machine-learning-approach)
4. [How We Transformed MSNA Burst Detection into an ML Task](#how-we-transformed-msna-burst-detection-into-an-ml-task)
5. [Explanation of the ML Model](#explanation-of-the-ml-model)
6. [Special Tweaks for Performance](#special-tweaks-for-performance)
7. [Explanation of Inter-Patient Performance Variability](#explanation-of-inter-patient-performance-variability)
8. [Usage Instructions](#usage-instructions)
9. [Project Structure](#project-structure)
10. [References & Acknowledgments](#references--acknowledgments)

---

## Introduction

Muscle Sympathetic Nerve Activity (MSNA) is a key physiological signal often examined to understand autonomic nervous system function. **Burst detection** in MSNA waveforms is crucial for clinical and research applications, allowing for the study of sympathetic outflow and its relationship with various health conditions.

In this project, we use **machine learning** to detect MSNA bursts, moving beyond simple peak-finding heuristics. By segmenting signals into ECG-defined cycles and extracting features, our model learns patterns distinguishing burst vs. non-burst cycles. The repository demonstrates:
- How data is loaded and standardized.
- How to craft input features.
- How to perform cross-validation and measure **F1** scores out-of-fold.
- How to ensure reproducibility and compliance with the project guidelines.

---

## Data and Preprocessing

1. **Data Source & Format**  
   - The data directory `./msna-data` contains CSV files of MSNA recordings and metadata.  
   - Each CSV has columns such as:
     - **Timestamp** (time in seconds)
     - **ECG** and **ECG Peaks** (R-wave indicators)
     - **Raw MSNA** and **Integrated MSNA**
     - **Burst** (ground-truth label indicating if a burst occurred)
   
2. **Filtering & Normalization**  
   - We apply a **Butterworth bandpass filter** (0.15–40 Hz) to remove irrelevant frequencies.  
   - We then **standardize** signals using 5th and 95th percentiles to set a common amplitude scale.

3. **ECG Cycle Segmentation**  
   - We locate R-peaks from the `ECG Peaks` column and define each cardiac cycle as the window between consecutive R-peaks.
   - Each cycle becomes one sample for the model, labeled by whether any burst occurred within that cycle.

---

## Machine Learning Approach

We choose a **Random Forest Classifier** (RF) for its balance of interpretability, speed, and performance. The pipeline:

1. **Feature Extraction** per ECG cycle:
   - **Max, Min, Mean, and Std of MSNA**  
   - **Cycle Length**  
   - **Relative Position of Max MSNA**  

2. **Model Training**:
   - **5-fold cross-validation** to produce out-of-fold predictions (and thus an unbiased F1 estimate).
   - We measure the **F1** score using the provided `msna_metric()` which aligns predictions with ground truth bursts per ECG cycle.

3. **Hyperparameter Tuning**:
   - A **grid search** over parameters like `n_estimators`, `max_depth`, `min_samples_split`, etc.

By default, the final chosen parameters are:
- `n_estimators = 200`
- `max_depth = None`
- `min_samples_split = 2`
- `min_samples_leaf = 1`
- `max_features = "sqrt"`

---

## How We Transformed MSNA Burst Detection into an ML Task

1. **Identify the Labels**  
   - The project data includes a binary column **Burst**, which indicates whether a burst was detected at each time sample.

2. **Segment into ECG Cycles**  
   - We leverage the **ECG Peaks** column to divide each recording into discrete cycles.  
   - Each cycle is labeled **1** if any burst occurred in that window, or **0** otherwise.

3. **Feature Engineering**  
   - For each cycle, we compute summary statistics (e.g., max, min, mean, etc.) of the **normalized MSNA** signal.  
   - This transforms the raw time-series into a structured feature vector.

4. **Supervised Classification**  
   - We treat each cycle as a training example with a binary label.
   - A Random Forest is trained to classify whether a burst occurs in that cycle.

5. **Cross-Validation & F1 Scoring**  
   - We perform 5-fold cross-validation: in each fold, the model is trained on 80% of the patients and tested on the remaining 20%.  
   - The **F1** score is computed out-of-fold via `msna_metric()`.

---

## Explanation of the ML Model

We use a **Random Forest Classifier**, an ensemble of decision trees:

- **Decision Trees** split features to minimize impurity (Gini or entropy), effectively capturing non-linear relationships in the data.
- **Ensemble Voting**: The forest aggregates predictions from multiple trees, reducing variance and improving generalization.
- **Interpretability**: Feature importance scores provide insights into which MSNA features are most predictive of bursts.

**Key Random Forest Hyperparameters**  
- **Number of Trees (`n_estimators`)**: More trees usually increase stability but at a higher computational cost.  
- **Max Depth (`max_depth`)**: Restricts how deep trees can grow, controlling overfitting.  
- **Min Samples Split / Min Samples Leaf**: Constraints to ensure each leaf or split has a minimum number of samples, reducing overfitting.  
- **Max Features**: Limits the number of features considered at each split, promoting diverse trees.

---

## Special Tweaks for Performance

1. **Feature Engineering**:  
   - **Cycle-level** features (amplitude, timing, variability) capture important MSNA burst characteristics.  
   - Normalization ensures consistent signal scales.

2. **Grid Search**:  
   - We systematically tested parameter combinations (e.g., tree count, max depth) to optimize the F1 score.  

3. **Filtering & Percentile Standardization**:  
   - Restricts frequencies to the range relevant for burst detection.  
   - Scales the signal to reduce patient-specific amplitude differences.

---

## Explanation of Inter-Patient Performance Variability

Five-fold cross-validation split the dataset by patient, yielding the following F1 scores per fold:

| Fold | Train Indices                  | Test Indices      | F1 Score  |
|-----:|:-------------------------------|:------------------|:----------|
| 1    | `[4..18]` (excluding 0–3)      | `[0,1,2,3]`       | 0.8716    |
| 2    | `[0..3, 8..18]` (excluding 4–7)| `[4,5,6,7]`       | 0.6111    |
| 3    | `[0..7, 12..18]` (excluding 8–11)| `[8,9,10,11]`    | 0.8375    |
| 4    | `[0..11, 16..18]` (excluding 12–15)| `[12,13,14,15]`| 0.7105    |
| 5    | `[0..15]` (excluding 16–18)    | `[16,17,18]`      | 0.8852    |

- **Mean F1**: ~0.80  
- **Std Dev**: ~0.08  

Variations can arise from patient-to-patient differences in MSNA amplitude, noise levels, or heart rate variability. Despite these differences, the random forest approach demonstrated robust performance overall, with most folds achieving strong F1 scores well above the baseline.
