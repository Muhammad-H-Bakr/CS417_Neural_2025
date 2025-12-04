# ✅ Universal Data Preprocessing

## For Regression, Binary Classification, and Multi-Class Classification

This guide covers **all essential preprocessing steps** for Machine Learning and Neural Networks.

---

## 1. Always Start Here (Initial Inspection)

```python
df.info()
df.describe()
df.isnull().sum()
df.nunique()
```

Classify each column:

| Type | Examples |
|------|------|
| Numerical | age, salary, area |
| Categorical (Nominal) | country, color |
| Categorical (Ordinal) | low, medium, high |
| Date/Time | timestamp |
| Text | reviews, comments |

---

## 2. Missing Values — Rules

### How much is missing?

| Missing % | Action |
|------|------|
| < 5% | Fill (Impute) |
| 5–30% | Try multiple methods |
| > 30% | Drop column |
| Missing in target | Drop row |

### What to fill with?

| Type | Best option |
|------|------|
| Numerical | Median |
| Categorical | Mode or "Unknown" |
| Ordinal | Mode or custom value |
| Text | "Unknown" |
| Important column | Create missing-flag + fill |

### Best method for Neural Networks

```python
df['Age_missing'] = df['Age'].isnull().astype(int)
df['Age'].fillna(df['Age'].median(), inplace=True)
```

---

## 3. Categorical Encoding (MOST IMPORTANT)

### Feature Columns (X)

| Situation | Use |
|------|------|
| Few categories (<15) | One-Hot |
| Many categories | Embedding / Target |
| Ordered categories | Manual mapping / OrdinalEncoder |
| Tree-based models | Label/Ordinal OK |

**Correct ordinal mapping:**

```python
size_map = {"Small":0, "Medium":1, "Large":2}
df['size'] = df['size'].map(size_map)
```

⚠ **Never use LabelEncoder on ordinal features — it sorts alphabetically.**

---

### Target Column (y)

| Task | Encoding |
|------|------|
| Regression | Leave as is |
| Binary Classification | 0 / 1 |
| Multi-class (NN) | One-Hot |
| Multi-class (Sklearn) | Integers |

For NN multi-class:

```python
y = LabelEncoder().fit_transform(y)
y = to_categorical(y)
```

---

## 4. Scaling / Normalization (CRITICAL)

### Features (X)

| Model | Scale? | Best choice |
|------|------|------|
| Neural Network | YES | StandardScaler |
| KNN / SVM | YES | StandardScaler |
| Linear / Logistic | YES | StandardScaler |
| Tree / Random Forest | NO | None |
| XGBoost | Optional | None / MinMax |
| Images | YES | MinMax (0–1) |

Standard:

```python
scaler = StandardScaler()
X = scaler.fit_transform(X)
```

---

### Target (y)

| Task | Scale? |
|------|------|
| Regression (NN) | YES |
| Regression (ML) | Optional |
| Classification | NO |

```python
y_scaler = StandardScaler()
y = y_scaler.fit_transform(y.reshape(-1,1))
```

---

## 5. Outlier Handling

Remove only if:

✅ Data error  
✅ Impossible values  
✅ > 4σ  

```python
from scipy.stats import zscore
z = np.abs(zscore(df['col']))
df = df[z < 3]
```

Do NOT remove rare but valid values.

---

## 6. Feature Engineering (Boosts performance)

✅ Reduce skew:

```python
df['salary'] = np.log1p(df['salary'])
```

✅ Date features:

```python
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
```

✅ Interaction:

```python
df['area_per_room'] = df['area'] / df['rooms']
```

---

## 7. Correct Order (No data leakage!)

```python
# Split first
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Fit scaler on train only
scaler.fit(X_train)

# Transform both
X_train = scaler.transform(X_train)
X_test  = scaler.transform(X_test)
```

❌ Never scale before split!

---

## 8. Loss Function (NN)

| Task | Output | Loss |
|------|------|------|
| Regression | Linear | mse |
| Binary | Sigmoid | binary_crossentropy |
| Multi-class (OneHot) | Softmax | categorical_crossentropy |
| Multi-class (labels) | Softmax | sparse_categorical_crossentropy |

---

## 9. Default Working Pipeline (90% of Problems)

```python
# Missing
df[num_cols] = df[num_cols].fillna(df[num_cols].median())
df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0])

# Encode
df = pd.get_dummies(df, columns=cat_cols)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

---

## 10. GOLDEN DECISION FLOW

Is it a category?
    Is there order?
        YES → Ordinal mapping
        NO  → One-hot

Is it numeric?
    Using NN/SVM/KNN?
        YES → Scale
        NO  → Leave

Is it target?
    Regression? → Scale
    Classification? → Encode
