# 🤖 Classical Machine Learning: From Scratch to scikit-learn

A hands-on learning portfolio covering the core **classical machine learning algorithms**. For most algorithms I first built the model **from scratch with NumPy** to understand the maths, then reproduced it with **scikit-learn** and compared the results. Each topic ends with a small mini-project.

> **About the data:** every notebook uses **synthetic datasets** generated with NumPy (fixed random seeds, so results are reproducible). The goal is to learn how algorithms behave when the true pattern is known, not to claim real-world business results. For analysis on real-world data, see the [related projects](#-related-projects) below.

---

## 🛠️ Tech Stack

`Python` · `NumPy` · `Pandas` · `scikit-learn` · `SciPy` · `Matplotlib` · `Seaborn` · `XGBoost` · `LightGBM` · `Optuna` · `Jupyter`

---

## 🗂️ Repository Guide

Files are numbered in the order I learned the topics. Notebooks marked **Project** end with a mini-project.

### 1. Foundations
| # | File | What it covers |
|---|---|---|
| 01 | [`01_preprocessing.ipynb`](01_preprocessing.ipynb) | Missing values (overall mean vs. group-wise mean), encoding, and a full `Pipeline` + `ColumnTransformer` (imputation, scaling, one-hot encoding) feeding a model |
| 02 | [`02_model_evaluation.py`](02_model_evaluation.py) | R², MAE, MSE and RMSE, calculated manually and with scikit-learn |
| 03 | [`03_model_improvement.ipynb`](03_model_improvement.ipynb) | K-fold and stratified K-fold cross-validation, `GridSearchCV`, `RandomizedSearchCV` and Optuna |

### 2. Regression
| # | File | What it covers |
|---|---|---|
| 04 | [`04_regression_models.ipynb`](04_regression_models.ipynb) | Linear, polynomial, Ridge, Lasso and Elastic Net regression (Ridge and Lasso also from scratch), decision tree, random forest and gradient boosting (scratch versions too), plus an introduction to XGBoost and LightGBM |
| 05 | [`05_project_exam_score_linear_regression.ipynb`](05_project_exam_score_linear_regression.ipynb) | **Project:** predict exam score from study hours. The scratch model learned w ≈ 4.9, b ≈ 40.4 against true values of 5 and 40 |
| 06 | [`06_project_salary_polynomial_regression.ipynb`](06_project_salary_polynomial_regression.ipynb) | **Project:** salary growth curve. Degrees 1–8 compared on train vs. test error to spot overfitting (best degree: 5) |
| 07 | [`07_project_salary_decision_tree.ipynb`](07_project_salary_decision_tree.ipynb) | **Project:** decision-tree regressor built from scratch (variance-based splits) and with scikit-learn, with a `max_depth` comparison |
| 08 | [`08_project_clv_random_forest.ipynb`](08_project_clv_random_forest.ipynb) | **Project:** customer lifetime value (CLV) prediction. Tree vs. forest across several seeds; tuning `max_features` and `n_estimators`; feature importance |
| 09 | [`09_project_clv_gradient_boosting.ipynb`](09_project_clv_gradient_boosting.ipynb) | **Project:** gradient boosting vs. random forest on the same CLV data, with an `n_estimators` × `learning_rate` search and a training-time comparison |

### 3. Classification
| # | File | What it covers |
|---|---|---|
| 10 | [`10_classification_models.ipynb`](10_classification_models.ipynb) | Logistic regression, KNN, Naive Bayes and SVM, each built from scratch and compared with scikit-learn, plus tree-based classifiers |
| 11 | [`11_logistic_regression.ipynb`](11_logistic_regression.ipynb) | Purchase prediction (time on site, pages visited, past purchases); stratified K-fold and PR-AUC for imbalanced data |
| 12 | [`12_project_loan_default_knn.ipynb`](12_project_loan_default_knn.ipynb) | **Project:** loan default risk. KNN from scratch with Euclidean and Manhattan distance, with vs. without feature scaling |
| 13 | [`13_project_spam_naive_bayes.ipynb`](13_project_spam_naive_bayes.ipynb) | **Project:** spam classifier. Gaussian Naive Bayes from scratch vs. scikit-learn |

### 4. Unsupervised Learning
| # | File | What it covers |
|---|---|---|
| 14 | [`14_unsupervised_learning.ipynb`](14_unsupervised_learning.ipynb) | K-Means from scratch, Gaussian Mixture Models, hierarchical clustering, DBSCAN (scratch and scikit-learn), PCA (scratch and scikit-learn) |
| 15 | [`15_project_customer_segmentation_kmeans.ipynb`](15_project_customer_segmentation_kmeans.ipynb) | **Project:** customer segmentation (annual spend × visit frequency). Elbow method and silhouette score selected **K = 3** (silhouette ≈ 0.68) |
| 16 | [`16_project_product_grouping_hierarchical.ipynb`](16_project_product_grouping_hierarchical.ipynb) | **Project:** product grouping by price and rating. Dendrograms with single / complete / ward linkage, compared with K-Means |
| 17 | [`17_project_sensor_anomaly_dbscan.ipynb`](17_project_sensor_anomaly_dbscan.ipynb) | **Project:** anomaly detection in sensor readings. `eps` chosen with a k-distance graph |
| 18 | [`18_project_pca_dimensionality_reduction.ipynb`](18_project_pca_dimensionality_reduction.ipynb) | **Project:** compressed 20 correlated features to **6 components** (95% of the variance), then trained a model on the reduced data |

---

## 💡 Key Concepts Practised

- **Bias–variance and overfitting:** train vs. test error curves (polynomial degree, tree depth)
- **Fair evaluation:** train/test split, cross-validation, several random seeds, fitting the scaler on training data only to avoid data leakage
- **Choosing hyperparameters:** elbow method, silhouette score, k-distance graph, grid and random search, Optuna
- **Algorithms from scratch:** linear, Ridge and Lasso regression, logistic regression, KNN, decision tree, random forest, gradient boosting, Gaussian Naive Bayes, SVM, K-Means, DBSCAN, PCA
- **Regularisation:** Ridge, Lasso and Elastic Net
- **Metrics:** R², MAE, MSE, RMSE, PR-AUC, silhouette score, explained variance

---

## ▶️ How to Run

```bash
git clone https://github.com/saniyanoori/classical-ML.git
cd classical-ML
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

No dataset downloads are needed because every notebook generates its own data.

---

## 🗺️ Roadmap

- [ ] Apply these models to a **real dataset** (for example customer churn or house prices) with a full cleaning → modelling → evaluation write-up
- [ ] Use `GridSearchCV` / `RandomizedSearchCV` inside a pipeline on that real dataset
- [ ] Add A/B testing and hypothesis-testing notebooks

---

## 🔗 Related Projects

- [Restaurant Data Analysis](https://github.com/saniyanoori/Restaurant-Data-Analysis): SQL + Python on restaurant sales
- [Media Survival Analysis](https://github.com/saniyanoori/Media_Survival_Analysis): Power BI + Python case study (Codebasics Resume Challenge 17)

## 👤 Author

**Saniya Noori** · [LinkedIn](https://www.linkedin.com/in/saniya-noori) · [GitHub](https://github.com/saniyanoori)
