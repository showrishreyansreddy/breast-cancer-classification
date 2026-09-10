# Explainable Machine Learning for Breast Cancer Recurrence Prediction

## Project Overview

This project applies machine learning and explainable AI techniques to the Wisconsin Breast Cancer Dataset stored as `WBCD_prognosis.csv`. The task is to predict **breast cancer recurrence (R) versus no recurrence (N)** from tumour-cell characteristics.

The notebook performs exploratory data analysis, data preprocessing, model training, model comparison, performance evaluation, feature importance analysis, and SHAP-based explainability.

Three supervised machine learning classifiers are compared:

- Logistic Regression
- Decision Tree
- Random Forest

The Random Forest model is also analysed using feature importance and SHAP to explain how individual features influence model predictions.

---

## Dataset

### Dataset Source

The dataset used in this project can be obtained from Kaggle:

**[Wisconsin Breast Cancer Dataset – Kaggle](https://www.kaggle.com/datasets/ucimachinelearning/wisconsin-breast-cancer-dataset)**

The Kaggle dataset page describes it as a dataset for **prognosis and recurrence prediction** and uses:

- `R` = **Recurrence**
- `N` = **No-Recurrence**

After downloading the dataset, place the CSV file in the same environment as the notebook and name it `WBCD_prognosis.csv` if necessary.

> **Note:** You may need a free Kaggle account to download the dataset.

The dataset used in the notebook contains:

- **569 records**
- **32 original columns**
- **1 ID column**
- **1 target column:** `diagnosis`
- **30 numerical input features**
- **No missing values**
- **No duplicate rows**

After removing the `id` column, the model uses **30 numerical features**.

The input features describe characteristics such as:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave points
- Symmetry
- Fractal dimension

For many characteristics, the dataset contains mean, standard-error (`_se`), and worst-value (`_worst`) measurements.

### Target Encoding

The dataset source defines the classes as:

- `N` = **No-Recurrence**
- `R` = **Recurrence**

The notebook maps the target variable as:

```python
df["diagnosis"] = df["diagnosis"].map({"R": 1, "N": 0})
```

Therefore:

- `0` = **No-Recurrence**
- `1` = **Recurrence**

Class distribution:

- `N` / No-Recurrence: **357 records**
- `R` / Recurrence: **212 records**

> **Important correction:** If any notebook chart currently labels class `0` as **Benign** and class `1` as **Malignant**, those chart labels should be changed to **No-Recurrence** and **Recurrence** respectively so that they agree with the dataset source.

---

## Project Workflow

The notebook follows the workflow below:

1. Install and import the required libraries.
2. Load `WBCD_prognosis.csv`.
3. Inspect the dataset structure and descriptive statistics.
4. Check missing values and duplicate records.
5. Remove the `id` column.
6. Encode the target variable.
7. Perform exploratory data analysis.
8. Separate features and target.
9. Split the data into training and testing sets.
10. Standardise features for Logistic Regression.
11. Train Logistic Regression, Decision Tree, and Random Forest models.
12. Evaluate the models using multiple classification metrics.
13. Generate confusion matrices.
14. Compare classification performance.
15. Plot ROC curves.
16. Analyse Random Forest feature importance.
17. Use SHAP for global and local model explanations.
18. Display sample Random Forest predictions with confidence values.

---

## Train-Test Split

The dataset is divided using an **80/20 stratified split**:

- Training records: **455**
- Testing records: **114**
- `random_state = 42`

Stratified sampling is used to preserve the target-class distribution.

---

## Machine Learning Models

### Logistic Regression

Logistic Regression is trained using standardised input features.

```python
LogisticRegression(random_state=42)
```

### Decision Tree

```python
DecisionTreeClassifier(random_state=42)
```

### Random Forest

The Random Forest is treated as the proposed ensemble model in the notebook.

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
```

---

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Precision
- Sensitivity / Recall
- Specificity
- F1 Score
- ROC-AUC
- False Negative Rate
- Confusion Matrix
- Classification Report
- ROC Curve

These metrics provide a broader assessment than accuracy alone.

---

## Model Results

| Model | Accuracy | Precision | Sensitivity | Specificity | F1 Score | ROC-AUC | False Negative Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9750 | 0.9286 | 0.9861 | 0.9512 | 0.9960 | 0.0714 |
| Decision Tree | 0.9298 | 0.9048 | 0.9048 | 0.9444 | 0.9048 | 0.9246 | 0.0952 |
| Random Forest | 0.9649 | 1.0000 | 0.9048 | 1.0000 | 0.9500 | 0.9942 | 0.0952 |

### Key Observations

- Logistic Regression and Random Forest both achieved approximately **96.49% accuracy**.
- Logistic Regression achieved the highest **ROC-AUC: 0.9960**.
- Random Forest achieved **1.0000 precision** and **1.0000 specificity** for the positive **recurrence** class under the notebook's encoding.
- Decision Tree achieved approximately **92.98% accuracy**.
- Logistic Regression recorded the lowest false-negative rate among the three models.

---

## Random Forest Feature Importance

The ten most important Random Forest features reported by the notebook are:

| Rank | Feature | Importance |
|---:|---|---:|
| 1 | `perimeter_worst` | 0.147912 |
| 2 | `area_worst` | 0.132269 |
| 3 | `concave points_worst` | 0.110114 |
| 4 | `concave points_mean` | 0.088200 |
| 5 | `radius_worst` | 0.085116 |
| 6 | `radius_mean` | 0.060554 |
| 7 | `perimeter_mean` | 0.060465 |
| 8 | `area_mean` | 0.044701 |
| 9 | `concavity_mean` | 0.044091 |
| 10 | `concavity_worst` | 0.032828 |

---

## Explainable AI with SHAP

SHAP is used to improve the interpretability of the Random Forest model.

The notebook creates:

- SHAP summary plot
- SHAP feature-importance bar plot
- Individual SHAP force plot

A TreeExplainer is created using:

```python
explainer = shap.TreeExplainer(rf)
shap_values = explainer.shap_values(X_test)
```

The global SHAP plots show which features have the greatest influence across model predictions, while the force plot explains the contribution of features for an individual sample.

---

## Technologies and Libraries

The project uses:

- Python
- Google Colab
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SHAP

---

## Installation

Install the required Python packages using:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap
```

---

## How to Run

1. Open `Showri_Final.ipynb` in Google Colab or Jupyter Notebook.
2. Make sure the dataset is available as:

```text
WBCD_prognosis.csv
```

3. In Google Colab, place the dataset at:

```text
/content/WBCD_prognosis.csv
```

4. Run the notebook cells in order from top to bottom.
5. Review the exploratory analysis, model evaluation, ROC curves, feature importance, and SHAP outputs.

---

## Suggested Repository Structure

```text
breast-cancer-ml/
│
├── Showri_Final.ipynb
├── WBCD_prognosis.csv
├── README.md
└── requirements.txt
```

If the dataset cannot be redistributed because of its licence or source conditions, do not upload it to the repository. Instead, provide a link to its authorised source.

---

## Example `requirements.txt`

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
shap
```

---

## Project Highlights

- End-to-end binary classification workflow
- Exploratory Data Analysis
- Data quality checking
- Stratified train-test split
- Feature standardisation
- Comparison of three machine learning algorithms
- Multiple classification evaluation metrics
- Confusion matrix analysis
- ROC and AUC comparison
- Random Forest feature importance
- SHAP-based global explainability
- SHAP-based individual prediction explanation

---

## Conclusion

This project demonstrates an end-to-end machine learning workflow for **breast cancer recurrence prediction**. Logistic Regression, Decision Tree, and Random Forest classifiers are trained and evaluated using multiple performance measures.

On the notebook's test split, Logistic Regression and Random Forest both achieve approximately **96.49% accuracy**, while Logistic Regression produces the highest ROC-AUC of **0.9960**. Random Forest additionally provides useful feature-importance information and is interpreted with SHAP to make its predictions more transparent.

The results demonstrate how predictive modelling can be combined with explainable AI techniques to support understanding of model behaviour.

---

## Important Note

This notebook is an academic/data-science project and should **not** be treated as a clinical diagnostic system. Model performance is based on the dataset and train-test split used in this notebook and does not establish clinical validity.

According to the linked dataset source, `R` represents **Recurrence** and `N` represents **No-Recurrence**. Any benign/malignant labels in notebook visualisations should therefore be corrected before presenting or submitting the project.

---

## Author

**Showri**

Machine Learning / Data Analytics Project
