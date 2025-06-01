
# ❤️ Heart Disease Detection using Boruta Feature Selection Algorithm

A Machine Learning project focused on predicting the presence of heart disease using various clinical features. This project utilizes the **Boruta algorithm** to select the most relevant features, improving model interpretability and accuracy.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Ideation & Brainstorming](#ideation--brainstorming)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Dataset](#dataset)
- [Modeling Process](#modeling-process)
- [Results](#results)
- [Usage](#usage)
- [Future Work](#future-work)
- [License](#license)

---

## 💡 Project Overview

Heart disease remains one of the leading causes of death globally. Early detection can save lives. This project applies supervised machine learning techniques, along with the Boruta feature selection algorithm, to identify critical indicators of heart disease from patient data.

---

## 🧠 Ideation & Brainstorming

### Problem Statement
- Existing models often include redundant or irrelevant features.
- Clinical data tends to be noisy and imbalanced.

### Goals
- Build a **robust classification model** for heart disease detection.
- Use **Boruta** to filter only the **most important features**.
- Ensure **high accuracy** with **low false negatives**.

### Brainstorming Tools
- Analyzed common ML techniques (Random Forest, Logistic Regression, SVM)
- Considered dimensionality reduction (PCA, LASSO) before settling on Boruta for its interpretability.

---

## 🚀 Features

- 🔍 **Boruta Feature Selection** – Wrapper method based on Random Forest
- 📊 **Data Visualization** – Histograms, correlation matrix, feature importances
- 🧠 **ML Models Implemented**: Logistic Regression, Random Forest, SVM, XGBoost
- 📈 **Model Evaluation**: Accuracy, Precision, Recall, ROC-AUC
- 💾 **Model Saving** – Exported trained model using `joblib`

---

## 🛠️ Tech Stack

- Python 3.9+
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- BorutaPy
- XGBoost
- Jupyter Notebook

---

## 📊 Dataset

- **Source**: [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)
- **Records**: 303 patients
- **Target Variable**: `target` (1: presence of heart disease, 0: absence)
- **Features**: Age, sex, chest pain type, blood pressure, cholesterol, etc.

---

## 🧪 Modeling Process

1. **Data Cleaning**: Removed nulls, encoded categorical values
2. **Exploratory Data Analysis**: Visualized distributions & correlations
3. **Feature Selection**: Used Boruta to select top features from dataset
4. **Train-Test Split**: 80/20 split
5. **Model Training**: Trained multiple models with cross-validation
6. **Evaluation**: Compared accuracy, F1-score, ROC-AUC

---

## ✅ Results

| Model            | Accuracy | ROC-AUC |
|------------------|----------|---------|
| Logistic Regression | 84%      | 0.88    |
| Random Forest       | 87%      | 0.91    |
| SVM                 | 85%      | 0.89    |
| XGBoost             | 89%      | 0.92    |

---

## 💻 Usage

### Clone Repository

```bash
git clone https://github.com/yourusername/Heart-Disease-Detection-using-Boruta-Feature-Selection-Algorithm.git
cd Heart-Disease-Detection-using-Boruta-Feature-Selection-Algorithm
```

### Run Notebook

Open `Heart_Disease_Boruta_Model.ipynb` using Jupyter Notebook or Google Colab.

### Predict

To use the saved model:

```python
import joblib
model = joblib.load("heart_disease_model.pkl")
sample = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
prediction = model.predict(sample)
print("Heart Disease Detected" if prediction[0] else "No Heart Disease")
```

---

## 🔮 Future Work

- Integrate with Flask/Django to deploy as a web app
- Add patient history and lifestyle metrics
- Use SHAP for better interpretability of model predictions

---


## 🙌 Acknowledgements

- UCI Machine Learning Repository
- BorutaPy Documentation
- Scikit-learn Contributors
- TowardsDataScience articles on feature selection
