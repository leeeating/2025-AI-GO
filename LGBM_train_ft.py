import json

import optuna
from optuna.samplers import TPESampler

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_recall_curve, log_loss


NUM = 1500
CUM = 15
folder = 'feat_v4'
RESULTS_FILE = f"./{folder}/optuna_gain_top{NUM}_cum{CUM}.json"


print("Loading data...")
df = pd.read_parquet("./Data/training_feature_v1.parquet")
test_df = pd.read_parquet("./testing/public_x.parquet")

# ========== 讀取特徵重要性 ==========
gain = pd.read_csv(filepath_or_buffer=f'./{folder}/gain.csv', index_col=0)
# gain = pd.read_csv(filepath_or_buffer=f'./{folder}/pos_shap_abs_mean.csv', index_col=0)

num_trails = gain.shape[1]

res = {}
for i in range(num_trails):
    rank = gain.iloc[:, i].nlargest(n=NUM).index
    for item in rank:
        if item not in res:
            res[item] = 0
        res[item] += 1
        
rank_cnt = pd.DataFrame(res.items(), columns=['feature', 'rank'])
selected_features = rank_cnt.query(f'rank >= {CUM}')['feature'].values

X = df.loc[:, selected_features]
y = df.iloc[:, -1]
print(f"------ X shape:{X.shape} ------")

# assert False

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42)

train_set = lgb.Dataset(X_train, label=y_train)
valid_set = lgb.Dataset(X_valid, label=y_valid)

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



def log_trial_result(study, trial):
    result = {
        "trial_number": trial.number,
        "f1": trial.values[0],
        # "log_loss": trial.values[1],
        "state": trial.state.name,
        "params": trial.params,
    }
    with open(RESULTS_FILE, "a+") as f:
        f.write(json.dumps(result, indent=4) + "\n")



def objective(trial):
    param = {
        'objective': 'binary',
        # 'metric': 'binary_logloss',
        'metric': ['f1', 'recall', 'precision', ],
        'verbosity': -1,
        'boosting_type': 'goss',
        'is_unbalance': True,
        
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, step=0.005),
        'num_leaves': trial.suggest_int('num_leaves', 5, 30, step=5),
        'max_depth': -1,
        
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0, step=0.1),  # GOSS 要設這個
        'top_rate': trial.suggest_float('top_rate', 0.1, 0.5, step=0.05),     # GOSS 專屬參數
        'other_rate': trial.suggest_float('other_rate', 0.1, 0.5, step=0.05), # GOSS 專屬參數

        'lambda_l2': trial.suggest_float('lambda_l2', 0, 5, step=1),

        'num_threads': 20,
    }

    # 使用 Optuna 的建議參數來訓練模型
    model = lgb.train(
            param,
            train_set=train_set,
            valid_sets=[valid_set],
            valid_names=['valid'],
            num_boost_round=10000,
            feval=[f1_eval, recall_eval, precision_eval,],
            callbacks=[
                lgb.early_stopping(stopping_rounds=1500, first_metric_only=True, verbose=True),
                lgb.log_evaluation(period=500, show_stdv=True),
                ],
        )
    preds = model.predict(X_valid)

    loss = log_loss(y_valid, preds)
    precs, recs, thresholds = precision_recall_curve(y_valid, preds)
    f1s = 2 * (precs * recs) / (precs + recs + 1e-10)
    f1s[np.isnan(f1s)] = 0
    best_idx = np.argmax(f1s)
    f1 = f1s[best_idx]
    prec = precs[best_idx]
    rec = recs[best_idx]
    best_thres = thresholds[best_idx]

    if (f1>=0.8) or (rec>=0.7 and prec>=0.87):
        test_prob = model.predict(data=test_df[selected_features])
        test_pred = (test_prob >= best_thres).astype(int)
        num_test_pos = test_pred.sum()
        print(f"Number of positive samples in test set: {num_test_pos}")
        model.save_model(f"./{folder}/fv4_{len(selected_features)}/fv4_{len(selected_features)}_{prec:.3f}_{rec:.3f}_{num_test_pos}.txt")


    print(f"Trial {trial.number} log loss: {loss:.5f} f1: {f1} pre: {prec} rec: {rec} \n")
    return f1


# study = optuna.create_study(sampler=NSGAIISampler(), directions=["maximize", "minimize"])
study = optuna.create_study(sampler=TPESampler(), direction="maximize", )

study.enqueue_trial(
params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'data_sample_strategy': 'goss',
    'metric': ['recall', 'f1', 'precision'],
    'is_unbalance': True,
    'verbosity': -1,

    'learning_rate': 0.04,
    'num_leaves': 15,
    'max_depth': -1,

    'feature_fraction': 1,
}
)

study.optimize(objective, n_trials=70, callbacks=[log_trial_result])