# Residue Contact Prediction — SVMs, Decision Trees, and Random Forests

## Overview
Binary classification to predict inter-helical residue contact in 
transmembrane proteins using SVMs, Decision Trees, and Random Forests.

**Course:** BINF 610 — Applied Machine Learning, University of Delaware

---

## Dataset
- Total examples: 10,000 residue pairs
- Feature vector: 375-dimensional (protein sequence co-evolution features)
- Column 1: ID
- Column 2: Ground truth label (1 = contact, 0 = no contact)
- Columns 3–377: Features
- Evaluation: 10-fold stratified cross validation

---

## Task 1 — Support Vector Machine (SVM)

### Part A — Without Feature Scaling (ROC-AUC)

| Kernel     | C=0.001 | C=0.01 | C=0.1  |
|------------|---------|--------|--------|
| Linear     | 0.8620  | 0.8716 | 0.8745 |
| Polynomial | 0.8268  | 0.8612 | 0.8764 |
| RBF        | 0.8295  | 0.8618 | 0.8852 |

**Best:** RBF kernel, C=0.1 → AUC = 0.8852

### Part B — With Feature Scaling (ROC-AUC)

| Kernel+C       | No Scale | MinMax | Standard |
|----------------|----------|--------|----------|
| linear_C0.001  | 0.8620   | 0.7993 | 0.8751   |
| linear_C0.01   | 0.8716   | 0.8575 | 0.8741   |
| linear_C0.1    | 0.8745   | 0.8744 | 0.8730   |
| poly_C0.001    | 0.8268   | 0.8179 | 0.6302   |
| poly_C0.01     | 0.8612   | 0.8679 | 0.7220   |
| poly_C0.1      | 0.8764   | 0.8754 | 0.8012   |
| rbf_C0.001     | 0.8295   | 0.7710 | 0.7880   |
| rbf_C0.01      | 0.8618   | 0.8259 | 0.8323   |
| rbf_C0.1       | 0.8852   | 0.8774 | 0.8803   |

### Key Findings — SVM
- Increasing C from 0.001 to 0.1 improved performance across all kernels
- Linear kernel was most stable across all C values and scaling methods
- Polynomial kernel was most sensitive to scaling — StandardScaler dropped 
  it from 0.8268 to 0.6302 at C=0.001
- Best overall: No Scaling + RBF + C=0.1 (AUC = 0.8852)
- Co-evolution features are already on comparable scales, making 
  additional scaling unnecessary or slightly harmful

---

## Task 2 — Decision Tree (CART Algorithm)

| Metric    | Full Tree (No Reg) | Max Depth = 6 |
|-----------|--------------------|---------------|
| Precision | 0.7167             | 0.8177        |
| Recall    | 0.7236             | 0.7160        |
| Accuracy  | 0.72               | 0.78          |
| F1-Score  | 0.72               | 0.76          |

### Key Findings — Decision Tree
- Full tree shows classic overfitting: memorizes training patterns 
  that don't generalize well
- Max depth = 6 improved precision by ~10% (0.7167 → 0.8177)
- Slight recall drop (0.7236 → 0.7160) reflects the classic 
  precision-recall trade-off from regularization
- Best configuration: Max depth = 6

---

## Task 3 — Random Forest (100 Trees)

| Metric    | No Contact | Contact | Overall |
|-----------|------------|---------|---------|
| Precision | 0.81       | 0.82    | 0.8199  |
| Recall    | 0.82       | 0.80    | 0.8014  |
| F1-Score  | 0.81       | 0.81    | 0.81    |
| Accuracy  | —          | —       | 0.81    |

### Top 5 Most Important Features

| Rank | Feature                  | Importance |
|------|--------------------------|------------|
| 1    | evfold_f_score_7         | 0.0348     |
| 2    | evfold_coupling_score_12 | 0.0307     |
| 3    | gdca_score_29            | 0.0307     |
| 4    | gdca_score_19            | 0.0302     |
| 5    | evfold_coupling_score_57 | 0.0277     |

All top 5 features belong to EVfold or GDCA co-evolution methods,
confirming these carry the strongest predictive signal for residue contact.

---

## Overall Conclusion

| Model                  | Best Metric        | Score  |
|------------------------|--------------------|--------|
| SVM (RBF, C=0.1)       | ROC-AUC            | 0.8852 |
| Decision Tree (depth=6)| Precision          | 0.8177 |
| Random Forest          | Balanced Precision/Recall | 0.81 |

- **Best AUC:** SVM with RBF kernel (0.8852)
- **Best balanced performance:** Random Forest (81% precision & recall)
- Random Forest outperformed Decision Tree significantly in recall 
  (0.7160 → 0.8014) confirming ensemble methods reduce variance better
  than single trees

---

## How to Run
```bash
pip install numpy pandas matplotlib scikit-learn
jupyter notebook residue_contact_classification.ipynb
```

---

## Files
- `residue_contact_classification.ipynb` — Full Jupyter notebook with 
  all code, plots, and analysis
- `data/` — Residue contact dataset (375 features, 10,000 examples)
