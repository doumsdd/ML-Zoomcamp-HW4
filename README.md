# ML Zoomcamp 2026 - Homework 4: Course Lead Scoring

## Summary of Questions & Answers

---

### Data Preparation

- **Missing values handling**: Categorical features replaced with `'NA'`, numerical features replaced with `0.0`
- **Data splits** (using `random_state=1`):
  - `df_full_train` / `df_test = train_test_split(df, test_size=0.2)` → full/train split (80/20)
  - `df_train` / `df_val = train_test_split(df_full_train, test_size=0.25)` → train/validation split (~75/25 of full)
  - Final sizes: `df_train` = 3000 rows, `df_val` = 1000 rows, `df_test` = 1000 rows

---

### Question 1: ROC AUC Feature Importance

**Task:** For each numerical variable (`lead_score`, `number_of_courses_viewed`, `interaction_count`, `annual_income`), compute the AUC using the variable as score and the target (`converted`) as ground truth on the training dataset. If AUC < 0.5, negate the variable.

**Results:**

| Variable | AUC | Inverted? |
|----------|-----|-----------|
| lead_score | **0.7882** | No |
| interaction_count | 0.7649 | No |
| number_of_courses_viewed | 0.7230 | No |
| annual_income | 0.6082 | No |

**Answer: `lead_score`** has the highest AUC at **0.7882**.

---

### Question 2: Training the Model

**Task:** Apply one-hot encoding using `DictVectorizer` and train logistic regression with `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)`. Compute AUC on the validation set.

**Code:**
```python
dv = DictVectorizer(sparse=False)
X_train_encoded = dv.fit_transform(X_train_dict)
X_val_encoded = dv.transform(X_val_dict)

model = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)
model.fit(X_train_encoded, y_train)
y_val_pred = model.predict_proba(X_val_encoded)[:, 1]
auc = roc_auc_score(y_val, y_val_pred)  # 0.7318
```

**Answer: 0.732**

---

### Question 3: Precision and Recall Intersection

**Task:** Evaluate precision and recall at thresholds from 0.0 to 1.0 (step 0.01). Find the threshold where the precision and recall curves intersect (smallest absolute difference, ignoring zero-value thresholds).

**Results at intersection:**

| Metric | Value |
|--------|-------|
| Threshold | **0.63** |
| Precision | 0.7328 |
| Recall | 0.7253 |
| Absolute Difference | 0.0075 |

**Answer: 0.63**

---

### Question 4: F1 Score Optimization

**Task:** Compute F1 score for all thresholds from 0.0 to 1.0 (step 0.01) and find the threshold that maximizes it.

Formula: `F1 = 2 * P * R / (P + R)`

**Results:**

| Metric | Value |
|--------|-------|
| Optimal Threshold | **0.41** |
| Max F1 Score | 0.7576 |
| Precision | 0.6190 |
| Recall | 0.9761 |

**Answer: 0.41**

---

### Question 5: 5-Fold Cross-Validation

**Task:** Use `KFold(n_splits=5, shuffle=True, random_state=1)` on `df_full_train`. Train `LogisticRegression(solver='liblinear', C=1.0, max_iter=1000)` per fold and evaluate AUC on validation folds. Compute the standard deviation of scores.

**Results per Fold:**

| Fold | AUC |
|------|-----|
| 1 | 0.7306 |
| 2 | 0.7296 |
| 3 | 0.7363 |
| 4 | 0.7412 |
| 5 | 0.7209 |

- **Mean AUC:** 0.7317
- **Std Dev (ddof=0):** **0.007**
- **Std Dev (ddof=1):** 0.008

**Answer: 0.007**

---

### Question 6: Hyperparameter Tuning

**Task:** Iterate over C values `[0.000001, 0.001, 1]` using 5-Fold CV. Compute mean AUC and std for each. Select the best C by highest mean AUC.

**Results:**

| C Value | Mean AUC | Std AUC |
|---------|----------|---------|
| 0.000001 | 0.617 | 0.019 |
| **0.001** | **0.749** | 0.010 |
| 1 | 0.732 | 0.007 |

**Answer: 0.001** (best mean AUC of 0.749)

---

## Code Reference

See `explore_data.py` for the complete implementation.
