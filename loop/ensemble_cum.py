import joblib
from itertools import product

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



# [223, 337, 639, 988, ]
NUMS = [2000, 1500, 1000, 500]
CUMS = [8, 9, 10, 11, 12, 13, 14, 15]
products = list(product(NUMS, CUMS))
num_trails = len(products)

pred_df = pd.DataFrame(index=test_df.index)
prob_df = pd.DataFrame(index=test_df.index)
np.random.seed(9527)
seeds = np.random.randint(0, 1000000, num_trails)

for idx, (num, cum) in enumerate(products):
    gain = pd.read_csv(filepath_or_buffer=f'../gain.csv', index_col=0)
    res = {}
    for i in range(15):
        rank = gain.iloc[:, i].nlargest(n=num).index
        for item in rank:
            if item not in res:
                res[item] = 0
            res[item] += 1
            
    rank_cnt = pd.DataFrame(res.items(), columns=['feature', 'rank'])
    selected_features = rank_cnt.query(f'rank >= {cum}')['feature']
    if len(selected_features) > 500:
        continue
    print(f"Selected {len(selected_features)} features")
    del gain

    X_train = train_df[selected_features]
    X_test = test_df[selected_features]
    y_train = train_df['飆股']
    print(f"Training Shape: {X_train.shape}, Test Shape: {X_test.shape}")

    print("Scaling data...")
    scaler = PowerTransformer(method='yeo-johnson', standardize=True)
    X_train_test = scaler.fit_transform(np.concatenate([X_train, X_test], axis=0))
    X_train_pt = X_train_test[:X_train.shape[0]]
    X_test_pt = X_train_test[X_train.shape[0]:]

    X_train, X_valid, y_train, y_valid = train_test_split(X_train_pt, y_train, test_size=0.1, random_state=seeds[idx], stratify=y_train)


    lgb_train_set = lgb.Dataset(X_train, y_train)
    lgb_valid_set = lgb.Dataset(X_valid, y_valid, reference=lgb_train_set)
    params = {
            'objective': 'binary',
            'boosting_type': 'gbdt',
            # 'data_sample_strategy': 'goss',
            'metric': ['recall', 'f1', 'precision', ],
            'metric': 'binary_logloss',
            'is_unbalance': True,
            'verbosity': -1,

            'learning_rate': 0.03,
            'num_leaves': 64,
            # 'min_data_in_leaf': 100,
            'max_depth': -1,

            'feature_fraction': 0.7,
            'bagging_fraction': 0.8,
            # 'top_rate': 0.25,
            # 'other_rate': 0.5,

            # 'lambda_l1': 1,
            # 'lambda_l2': 1,

            'n_jobs': 30,
            'seed': seeds[idx],
            }
    
    
    model = lgb.train(
        params,
        train_set=lgb_train_set,
        valid_sets=[lgb_valid_set],
        valid_names=['valid'],
        num_boost_round=10000,
        feval=[eval_all,],
        callbacks=[
            # lgb.early_stopping(stopping_rounds=3000, verbose=True, first_metric_only=True),
            lgb.log_evaluation(period=500),
            # lgb.reset_parameter(learning_rate=dynamic_lr),
            ])
    
    valid_prob = model.predict(X_valid)
    prec, recall, thresholds = precision_recall_curve(y_valid, valid_prob)
    f1s = 2 * (prec * recall) / (prec + recall + 1e-10)
    f1s[np.isnan(f1s)] = 0
    best_idx = np.argmax(f1s)
    best_threshold = thresholds[best_idx]
    print(f"Best threshold: {best_threshold}")
    
    lgb_all_set = lgb.Dataset(np.concatenate([X_train, X_valid], axis=0), np.concatenate([y_train, y_valid], axis=0))
    second_model = lgb.train(
        params,
        train_set=lgb_all_set,
        num_boost_round=10000,
        feval=[eval_all,],
        callbacks=[
            lgb.log_evaluation(period=1000),
            ])

    
    test_prob = second_model.predict(X_test_pt)
    test_pred = (test_prob >= best_threshold).astype(int)
    public_pos = np.sum(test_pred[:25108])
    private_pos = np.sum(test_pred[25108:])
    print(f"Public pos: {public_pos}, Private pos: {private_pos}")

    pred_df[idx] = test_pred
    prob_df[idx] = test_prob
    print(f"Model {idx} finished.")

    save_dict = {
        'model': second_model,
        'scaler': scaler,
        'selected_features': selected_features,
        'best_threshold': best_threshold,
        'pos': (public_pos, private_pos),
        'eval': (prec, recall, thresholds),
    }
    joblib.dump(save_dict, f'./ense_result/trail_{idx}.pkl')
    print(f"Model {idx} saved.")

pred_df.to_csv('ensemble_test_pred.csv', index=False)
prob_df.to_csv('ensemble_test_prob.csv', index=False)