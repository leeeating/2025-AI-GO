import json
import joblib

import lightgbm as lgb
import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler


from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score

print("Loading data...")
train_df = pd.read_parquet("../../Data/training_feature_v1.parquet")
test_df = pd.read_parquet("../../testing/X_all.parquet")
tmpl = pd.read_csv(filepath_or_buffer='../../testing/submission_template_public_and_private.csv')

NUM = 1000
CUM = 10
# ========== 讀取特徵重要性 ==========
gain = pd.read_csv(filepath_or_buffer=f'../gain.csv', index_col=0)
num_trails = gain.shape[1]

res = {}
for i in range(num_trails):
    rank = gain.iloc[:, i].nlargest(n=NUM).index
    for item in rank:
        if item not in res:
            res[item] = 0
        res[item] += 1
        
rank_cnt = pd.DataFrame(res.items(), columns=['feature', 'rank'])
selected_features = rank_cnt.query(f'rank >= {CUM}')['feature']
print(f"Selected {len(selected_features)} features")

X_train = train_df[selected_features]
X_test = test_df[selected_features]
y_train = train_df['飆股']

scaler = PowerTransformer(method='yeo-johnson', standardize=True)
X_train_test = scaler.fit_transform(np.concatenate([X_train, X_test], axis=0))
X_train_pt = X_train_test[:X_train.shape[0]]
X_test_pt = X_train_test[X_train.shape[0]:]


def eval_all(preds, dtrain):
    y_true = dtrain.get_label()
    prec, recall, thresholds = precision_recall_curve(y_true, preds)
    f1s = 2 * (prec * recall) / (prec + recall + 1e-10)
    f1s[np.isnan(f1s)] = 0
    best_idx = np.argmax(f1s)
    return [
            ('f1', f1s[best_idx], True),
            ('precision', prec[best_idx], True),
            ('recall', recall[best_idx], True),
            ]

def log_trial_result(study, trial):
    result = {
        "trial_number": trial.number,
        "f1": trial.values[0],
        # "log_loss": trial.values[1],
        "state": trial.state.name,
        "params": trial.params,
    }
    with open(f"./optuna_top{NUM}_cum{CUM}.json", "a+") as f:
        f.write(json.dumps(result, indent=4) + "\n")



trial_f1 = []
trial_prec = []
trial_recall = []
def objective(trial):
    global trial_f1, trial_prec, trial_recall
    print(f"Trial {trial.number} started\n")
    params = {
        'objective': 'binary',
        # 'metric': 'binary_logloss',
        'metric': ['f1', 'recall', 'precision', ],
        'verbosity': -1,
        'boosting_type': 'goss',
        'is_unbalance': True,
        
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.07, step=0.01),
        'num_leaves': trial.suggest_int('num_leaves', 50, 260, step=10),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 100, 500, step=20),
        'max_depth': -1,
        
        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0, step=0.05),  # GOSS 要設這個
        'top_rate': trial.suggest_float('top_rate', 0.1, 0.5, step=0.05),     # GOSS 專屬參數
        'other_rate': trial.suggest_float('other_rate', 0.1, 0.5, step=0.05), # GOSS 專屬參數

        'num_threads': 20,
    }


    f1_list = []
    prec_list = []
    recall_list = []
    stf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(stf.split(X_train_pt, y_train)):

        X_train_fold, X_val_fold = X_train_pt[train_idx], X_train_pt[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        dtrain = lgb.Dataset(X_train_fold, label=y_train_fold)
        dval = lgb.Dataset(X_val_fold, label=y_val_fold, reference=dtrain)

        model = lgb.train(
            params,
            train_set=dtrain,
            valid_sets=[dval],
            valid_names=['valid'],
            num_boost_round=10000,
            feval=[eval_all,],
            callbacks=[
                lgb.early_stopping(stopping_rounds=3000, first_metric_only=True, verbose=True),
                lgb.log_evaluation(period=1000),
                # lgb.reset_parameter(learning_rate=dynamic_lr),
                ], )

        valid_prob = model.predict(X_val_fold)
        prec, recall, thresholds = precision_recall_curve(y_val_fold, valid_prob)
        f1s = 2 * (prec * recall) / (prec + recall + 1e-10)
        f1s[np.isnan(f1s)] = 0
        best_idx = np.argmax(f1s)
        f1_list.append(f1s[best_idx])
        prec_list.append(prec[best_idx])
        recall_list.append(recall[best_idx])
        print(f"Fold {fold} - f1: {f1s[best_idx]}, precision: {prec[best_idx]}, recall: {recall[best_idx]}")
    
        if (f1s[best_idx] > 0.83) or (prec[best_idx] > 0.95) or (recall[best_idx] > 0.78):

            test_prob = model.predict(X_test_pt)
            best_threshold = thresholds[best_idx]
            test_pred = (test_prob >= best_threshold).astype(int)
            pub_pos = test_pred[:25108]
            pri_pos = test_pred[25108:]

            test_pred_df = pd.DataFrame({
                'id': tmpl['ID'],
                'pred': test_pred,
                'prob': test_prob,
            })

            save_dict = {
                'model': model,
                'params': params,
                'threshold': best_threshold,
                'public_pos': pub_pos,
                'private_pos': pri_pos,
                'eval_results': {
                    'f1': f1s[best_idx],
                    'precision': prec[best_idx],
                    'recall': recall[best_idx],
                }
            }
            joblib.dump(save_dict, f'./model/{f1s[best_idx]*10000:.0f}_{prec[best_idx]*10000:.0f}_{recall[best_idx]*10000:.0f}.pkl')
            test_pred_df.to_csv(f"./smt/{f1s[best_idx]*10000:.0f}_{prec[best_idx]*10000:.0f}_{recall[best_idx]*10000:.0f}.csv", index=False)
    
    avg_f1 = np.mean(f1_list)
    avg_prec = np.mean(prec_list)
    avg_recall = np.mean(recall_list)

    trial_f1.append(avg_f1)
    trial_prec.append(avg_prec)
    trial_recall.append(avg_recall)
    print(f"Trial {trial.number} - f1: {avg_f1}, precision: {avg_prec}, recall: {avg_recall}")
    print(f"Trial {trial.number} finished\n")

    return avg_f1


study = optuna.create_study(sampler=TPESampler(), direction="maximize", )

study.enqueue_trial(
        params = {
                'objective': 'binary',
                'boosting_type': 'gbdt',
                'data_sample_strategy': 'goss',
                'metric': ['recall', 'f1', 'precision', ],
                # 'metric': 'binary_logloss',
                'is_unbalance': True,
                'verbosity': -1,

                'learning_rate': 0.03,
                'num_leaves': 128,
                'max_depth': -1,
                'min_data_in_leaf':300,
                'feature_pre_filter': False,

                'feature_fraction': 0.7,
                # 'bagging_fraction': 0.75,
                'top_rate': 0.25,
                'other_rate': 0.5,

                'n_jobs': 30,
            })

study.optimize(objective, n_trials=100, callbacks=[log_trial_result])

eval_df = pd.DataFrame({
    'trial_f1': trial_f1,
    'trial_prec': trial_prec,
    'trial_recall': trial_recall,
})

eval_df.to_csv("./optuna_eval.csv", index=False)

