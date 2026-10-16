import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

url = 'https://raw.githubusercontent.com/DataTalksClub/machine-learning-zoomcamp/main/cohorts/2026/data/course_lead_scoring_2026.csv'
df = pd.read_csv(url)

categorical_cols = ['lead_source', 'industry', 'employment_status', 'location']
numerical_cols = ['annual_income', 'number_of_courses_viewed', 'interaction_count', 'lead_score']

for col in categorical_cols:
    df[col] = df[col].fillna('NA')
for col in numerical_cols:
    df[col] = df[col].fillna(0.0)

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

features = ['lead_source', 'industry', 'employment_status', 'location',
            'annual_income', 'number_of_courses_viewed', 'interaction_count', 'lead_score']

# --- Question 6: Hyperparameter Tuning ---
C_values = [0.000001, 0.001, 1]
kfold = KFold(n_splits=5, shuffle=True, random_state=1)

print('--- Question 6: Hyperparameter Tuning ---')
for C in C_values:
    auc_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(df_full_train)):
        df_train_fold = df_full_train.iloc[train_idx]
        df_val_fold = df_full_train.iloc[val_idx]

        X_train_fold = df_train_fold[features].to_dict(orient='records')
        X_val_fold = df_val_fold[features].to_dict(orient='records')
        y_train_fold = df_train_fold['converted'].values
        y_val_fold = df_val_fold['converted'].values

        dv = DictVectorizer(sparse=False)
        X_train_encoded = dv.fit_transform(X_train_fold)
        X_val_encoded = dv.transform(X_val_fold)

        model = LogisticRegression(solver='liblinear', C=C, max_iter=1000)
        model.fit(X_train_encoded, y_train_fold)

        y_val_pred_proba = model.predict_proba(X_val_encoded)[:, 1]
        auc = roc_auc_score(y_val_fold, y_val_pred_proba)
        auc_scores.append(auc)

    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores, ddof=0)
    print(f'C = {C}: Mean AUC = {mean_auc:.3f}, Std AUC = {std_auc:.3f}')


