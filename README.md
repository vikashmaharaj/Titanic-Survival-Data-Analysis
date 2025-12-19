# Titanic Survival Data Analysis

## 1. API Key Setup & Dataset Source

**First step (very important): Kaggle API authentication**

To download the Titanic dataset programmatically, a Kaggle API key is required.

### Steps to get Kaggle API Key:

1. Go to **[https://www.kaggle.com/](https://www.kaggle.com/)**
2. Login → Profile → **Account**
3. Scroll to **API** section
4. Click **Create New API Token**
5. A file named `kaggle.json` will be downloaded

Place `kaggle.json` in the following location:

* **Windows:** `C:/Users/<username>/.kaggle/kaggle.json`
* **Linux / Colab:** `/root/.kaggle/kaggle.json`

Give proper permission:

```bash
chmod 600 kaggle.json
```

### Dataset Link:

Titanic – Machine Learning from Disaster (Kaggle)

```
https://www.kaggle.com/c/titanic
```

---

## 2. Problem Statement

**Question:**

> Given passenger information (age, gender, ticket class, fare, etc.), can we predict whether a passenger survived the Titanic disaster?

This is a **Supervised Machine Learning – Classification** problem.

* **Input:** Passenger features
* **Output:** Survival (0 = No, 1 = Yes)

---

## 3. Libraries Used

```python
import pandas as pd
import opendatasets as od
```

* **pandas** → Data manipulation and analysis
* **opendatasets** → Download Kaggle datasets using API

---

## 4. Step-by-Step Problem Solving Approach

### Step 1: Download the Dataset

```python
od.download("https://www.kaggle.com/c/titanic")
```

✔ Automatically downloads Titanic dataset from Kaggle

---

### Step 2: Load the Dataset

```python
df = pd.read_csv("/content/titanic/train.csv")
```

✔ Reads training data into a DataFrame

---

### Step 3: Initial Data Exploration

#### View the data

```python
df.head()
df.tail()
```

✔ Understand structure and sample records

---

### Step 4: Understand Dataset Information

```python
df.info()
```

✔ Identifies:

* Total rows & columns
* Data types
* Missing values

---

### Step 5: Statistical Summary

```python
df.describe(include="all")
```

✔ Shows:

* Mean, median, min, max
* Unique values for categorical data

---

### Step 6: Missing Value Analysis

```python
df.isnull().sum()
```

✔ Detects columns with missing data (Age, Cabin, Embarked)

---

## 5. Questions Solved During Analysis

1. How many passengers survived vs not survived?
2. Does gender affect survival?
3. Does passenger class affect survival?
4. Which columns contain missing values?
5. What type of data cleaning is required?

Each question is answered using **data inspection and statistical analysis** before model building.

---

## 6. Why This Step-by-Step Approach?

✔ Understand data before modeling
✔ Avoid garbage-in-garbage-out problem
✔ Identify missing and categorical features
✔ Prepare data for feature engineering

---

## 7. Final Outcome

* Dataset successfully downloaded using Kaggle API
* Data loaded and explored
* Missing values identified
* Clear understanding of the survival prediction problem

This analysis forms the **foundation for building a machine learning model** such as Logistic Regression or Decision Tree.

---

## 8. Next Steps (Future Work)

* Data cleaning & imputation
* Feature encoding
* Model training
* Model evaluation

---

