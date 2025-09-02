import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_curve, log_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


print("Loading data...")
df = pd.read_parquet("./Data/training_feature_v1.parquet")
y = df.iloc[:, -1]

X = df.iloc[:, 1:-1]
mis_rate = X.isna().sum() / X.shape[0]
sel_feat = mis_rate[mis_rate<0.2].index
X = X[sel_feat]
print(f"X shape: {X.shape}")


std_x = StandardScaler().fit_transform(X)
X = pd.DataFrame(std_x, columns=X.columns)
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")


# assert False
def f1_eval(preds, dtrain):
    y_true = dtrain.get_label()
    prec, recall, thresholds = precision_recall_curve(y_true, preds)
    f1s = 2 * (prec * recall) / (prec + recall + 1e-10)
    f1s[np.isnan(f1s)] = 0
    best_idx = np.argmax(f1s)
    best_f1 = f1s[best_idx]
    return 'f1', best_f1, True

def precision_eval(preds, dtrain):
    y_true = dtrain.get_label()
    prec, recall, thresholds = precision_recall_curve(y_true, preds)
    f1s = 2 * (prec * recall) / (prec + recall + 1e-10)
    f1s[np.isnan(f1s)] = 0
    best_idx = np.argmax(f1s)
    best_prec = prec[best_idx]
    return 'precision', best_prec, True

def recall_eval(preds, dtrain):
    y_true = dtrain.get_label()
    prec, recall, thresholds = precision_recall_curve(y_true, preds)
    f1s = 2 * (prec * recall) / (prec + recall + 1e-10)
    f1s[np.isnan(f1s)] = 0
    best_idx = np.argmax(f1s)
    best_recall = recall[best_idx]
    return 'recall', best_recall, True


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


# ====== 多次 seed gain importance + SHAP 整合 ======
np.random.seed(105)
seeds = np.random.randint(0, 10000, 15)


feature_gain_all = np.zeros((X.shape[1], len(seeds)))
for idx, seed in enumerate(seeds):
    print(f"================= {idx+1}/{len(seeds)}  Seed {seed} =================")
    train_set = lgb.Dataset(X_train, label=y_train)
    valid_set = lgb.Dataset(X_val, label=y_val)

    params = {
        'objective': 'binary',
        'boosting_type': 'gbdt',
        # 'data_sample_strategy': 'goss',
        'metric': ['binary_logloss'],
        'is_unbalance': True,
        'verbosity': 1,
        'learning_rate': 0.05,
        # 'num_leaves': 128,
        'max_depth': 10,

        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,

        'lambda_l1': 1.0,
        
        'n_jobs': 40,
        'seed': seed,
        # 'zero_as_missing': True,
    }

    model = lgb.train(
        params,
        train_set=train_set,
        valid_sets=[valid_set],
        feval=f1_eval,
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(stopping_rounds=400, verbose=True),
            lgb.log_evaluation(period=10)]
    )

    model.save_model(f"./muti_seed_model/std_fi/lgbm_seed{seed}.txt")

    # 累加 gain
    gain = model.feature_importance(importance_type='gain')
    feature_gain_all[:, idx] = gain

print("Feature gain importance shape:", feature_gain_all.shape)
print('Saving feature gain importance...')
feature_gain_df = pd.DataFrame(feature_gain_all, columns=X.columns)
feature_gain_df.to_csv("./std_fi/gain.csv", index=False)
