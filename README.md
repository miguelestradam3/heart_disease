# Support Vector Machines in Python: Heart Disease Classification

## Overview
This project demonstrates how to build, evaluate, and interpret a **Support Vector Machine (SVM)** classifier from scratch using **scikit-learn** and the **Radial Basis Function (RBF)** kernel. The model utilizes continuous and categorical clinical data from the **UCI Machine Learning Repository** to predict whether or not a patient has **heart disease**.

Support vector machines are powerful machine learning models particularly well-suited for classification tasks where obtaining high accuracy takes precedence over understanding the underlying decision mechanisms, and they perform exceptionally well with relatively small datasets.

---

## What You Will Learn
* **Data Import & Inspection:** Loading raw data files into pandas dataframes and assigning meaningful feature names.
* **Handling Missing Data:** Identifying surrogate missing values (such as `?`) and deciding whether to drop rows or impute values.
* **Data Formatting:** Splitting datasets into dependent and independent variables, and binarizing multi-level classification targets.
* **One-Hot Encoding:** Converting multi-category categorical variables into binary indicator columns using pandas.
* **Centering and Scaling:** Standardizing features separately for training and testing sets to prevent data leakage before feeding data into an RBF kernel.
* **Model Training & Optimization:** Building a preliminary support vector classifier and optimizing parameters using cross-validation (`GridSearchCV`).

---

## Requirements & Dependencies
Ensure you have **Python 3** installed along with the following minimum library versions:
* `pandas` >= 0.25.1
* `numpy` >= 1.17.2
* `scikit-learn` >= 0.22.1
* `matplotlib` / `seaborn` (for data visualization)

If you use Anaconda, you can manage packages via `conda install scikit-learn=0.22.1` or update your environment packages accordingly.

---

## Dataset
This project uses the **Heart Disease Dataset** from the **UCI Machine Learning Repository** (`processed.cleveland.data`). Features include:
* **age**: Patient age in years
* **sex**: Biological sex (0 = female, 1 = male)
* **cp**: Chest pain type (1-4)
* **restbp**: Resting blood pressure in mm Hg
* **chol**: Serum cholesterol in mg/dl
* **fbs**: Fasting blood sugar
* **restecg**: Resting electrocardiographic results
* **thalach**: Maximum heart rate achieved
* **exang**: Exercise-induced angina
* **oldpeak**: ST depression induced by exercise relative to rest
* **slope**: Slope of the peak exercise ST segment
* **ca**: Number of major vessels colored by fluoroscopy
* **thal**: Thallium heart scan diagnosis
* **hd**: Diagnosis of heart disease (predicted target attribute)

---

## Project Workflow

1. **Import Modules:** Load essential data manipulation, visualization, and machine learning packages (`pandas`, `numpy`, `matplotlib`, `seaborn`, and `sklearn`).
2. **Load Data:** Read the dataset into a pandas dataframe and assign standard clinical feature names.
3. **Clean Missing Data:** Detect missing entries represented as `?` in columns like `ca` and `thal`, removing affected rows safely since they constitute a small fraction (~2%) of the dataset.
4. **Split Variables:** Separate features (`X`) from the target diagnosis (`y`), mapping multi-level heart disease diagnoses into a binary classification problem (0 = no disease, 1 = disease present).
5. **One-Hot Encoding:** Transform categorical variables with more than two categories (`cp`, `restecg`, `slope`, `thal`) into binary indicator columns using `pd.get_dummies()`.
6. **Scale & Center:** Split data into training and testing subsets (`train_test_split`) and scale features separately to avoid data leakage.
7. **Train SVM:** Initialize and train the preliminary Support Vector Classifier using the RBF kernel (`SVC`).

---

## How to Run
Open the Jupyter Notebook or Python script containing the workflow in your preferred development environment and execute the cells sequentially. 

> **Tip:** Pause before running each major code block to understand the transformations taking place, and experiment with alternative parameters to deepen your understanding of SVM behavior!
