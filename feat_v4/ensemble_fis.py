import joblib

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split

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

print("Loading data...")
train_df = pd.read_parquet("../../Data/training_feature_v1.parquet")
test_df = pd.read_parquet("../../testing/X_all.parquet")
sudo1_df= pd.read_csv('../../testing/guess136.csv')

tmpl = pd.read_csv(filepath_or_buffer='../../testing/submission_template_public_and_private.csv')


NUM = 2000
CUM = 10

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
del gain

assert False

np.random.seed(9527)
seeds = np.random.randint(0, 10000, size=20)

pred_df = pd.DataFrame(index=test_df.index, columns=range(num_trails))
prob_df = pd.DataFrame(index=test_df.index, columns=range(num_trails))
for i in range(num_trails):

    X_train = train_df[selected_features]
    X_test = test_df[selected_features]
    y_train = train_df['飆股']
    print(f"Training Shape: {X_train.shape}, Test Shape: {X_test.shape}, Seed: {seeds[i]}")

    print("Scaling data...")
    scaler = PowerTransformer(method='yeo-johnson', standardize=True)
    X_train_test = scaler.fit_transform(np.concatenate([X_train, X_test], axis=0))
    X_train_pt = X_train_test[:X_train.shape[0]]
    X_test_pt = X_train_test[X_train.shape[0]:]

    X_train, X_valid, y_train, y_valid = train_test_split(X_train_pt, y_train, test_size=0.9, random_state=seeds[i], stratify=y_train)


    lgb_train_set = lgb.Dataset(X_train, y_train)
    lgb_valid_set = lgb.Dataset(X_valid, y_valid, reference=lgb_train_set)
    params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            # 'data_sample_strategy': 'goss',
            # 'metric': ['recall', 'f1', 'precision', ],
            'metric': 'binary_logloss',
            'is_unbalance': True,
            'verbosity': -1,

            'learning_rate': 0.03,
            'num_leaves': 128,
            'min_data_in_leaf': 100,
            'max_depth': -1,

            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,

            'lambda_l1': 5,
            'lambda_l2': 1,

            'n_jobs': 30,
            'seed': seeds[i],
            }
    
    
    model = lgb.train(
        params,
        train_set=lgb_train_set,
        valid_sets=[lgb_valid_set],
        valid_names=['valid'],
        num_boost_round=10000,
        feval=[eval_all,],
        callbacks=[
            lgb.log_evaluation(period=500),
            # lgb.reset_parameter(learning_rate=dynamic_lr),
            ])
    

    feature_importance = model.feature_importance(importance_type='gain')
