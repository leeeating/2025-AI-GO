import lightgbm as lgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold

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
train_df = pd.read_parquet("../Data/training_feature_v1.parquet")
test_df = pd.read_parquet("../testing/X_all.parquet")
sudo1_df= pd.read_csv('../testing/guess136.csv')

tmpl = pd.read_csv('../testing/submission_template_public_and_private.csv')


NUM = 500
CUM = 10

gain = pd.read_csv(filepath_or_buffer=f'../feat_v4/gain.csv', index_col=0)
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




X_train = train_df[selected_features]
X_test = test_df[selected_features]
y_train = train_df['飆股']

scaler = PowerTransformer(method='yeo-johnson', standardize=True)
X_train_test = scaler.fit_transform(np.concatenate([X_train, X_test], axis=0))
X_train_pt = X_train_test[:X_train.shape[0]]
X_test_pt = X_train_test[X_train.shape[0]:]
np.random.seed(1111)
seeds = np.random.randint(0, 10000, size=10)
num_splits = 10

# known test data
zero_list = [
    6722, 8831, 11999, 7555, 22570, 6898, 20029, 14120,
    14467, 4160, 12519, 8484, 9251, 18104, 18850, 15676, 16995,
    23508, 15749, 24661, 19364, 13181, 5418, 7613, 23976, 780,
    23557, 10288, 5375, 17790, 3479,2731,7752,6833,22045,23492,18305,14120,4328]

imp_zero = np.array(zero_list) - 1
imp_one = np.where(sudo1_df['飆股']==1)[0]
concat_test_idx = np.concatenate([imp_zero, imp_one])
print(f"Length of test index: {len(concat_test_idx)}")

X_test_known = X_test_pt[concat_test_idx]
y_test_known = np.concatenate([np.zeros(len(imp_zero)), np.ones(len(imp_one))])


# result variables
thres_rec = np.zeros((len(seeds), num_splits))
f1_rec = np.zeros((len(seeds), num_splits))
prec_rec = np.zeros((len(seeds), num_splits))
recall_rec = np.zeros((len(seeds), num_splits))
val_wrong_cnt = np.zeros((X_train.shape[0], ))
val_prob_metrics = np.zeros((X_train.shape[0], len(seeds)))
val_pred_metrics = np.zeros((X_train.shape[0], len(seeds)))

test_prob_df = pd.DataFrame(index=tmpl['ID'])
test_pred_df = pd.DataFrame(index=tmpl['ID'])


for idx, seed in enumerate(seeds):
    stf = StratifiedKFold(n_splits=num_splits, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(stf.split(X_train, y_train)):
        print(f"Iteration {idx+1}, fold {fold+1}")

        X_train_fold, X_val_fold = X_train_pt[train_idx], X_train_pt[val_idx]
        y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # LightGBM
        lgb_train_set = lgb.Dataset(X_train_fold, label=y_train_fold)
        lgb_valid_set = lgb.Dataset(X_val_fold, label=y_val_fold, reference=lgb_train_set)

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
            }
        model = lgb.train(
            params,
            train_set=lgb_train_set,
            valid_sets=[lgb_valid_set],
            valid_names=['valid'],
            num_boost_round=10000,
            feval=[eval_all,],
            callbacks=[
                lgb.early_stopping(stopping_rounds=3000, first_metric_only=True, verbose=True),
                lgb.log_evaluation(period=1000),
                # lgb.reset_parameter(learning_rate=dynamic_lr),
                ], )
        
        val_prob = model.predict(X_val_fold, num_iteration=model.best_iteration)


        prec, recall, thresholds = precision_recall_curve(y_val_fold, val_prob)
        f1s = 2 * (prec * recall) / (prec + recall + 1e-10)
        f1s[np.isnan(f1s)] = 0
        best_idx = np.argmax(f1s)
        best_threshold = thresholds[best_idx]
        thres_rec[idx, fold] = best_threshold
        f1_rec[idx, fold] = f1s[best_idx]
        prec_rec[idx, fold] = prec[best_idx]
        recall_rec[idx, fold] = recall[best_idx]
        print(f"Threshold: {best_threshold:.4f}, f1: {f1s[best_idx]:.4f}, precision: {prec[best_idx]:.4f}, recall: {recall[best_idx]:.4f}")


        val_pred = (val_prob >= best_threshold).astype(int)
        wrong_idx = np.where(val_pred != y_val_fold.values.reshape(-1))[0]
        val_wrong_cnt[val_idx][wrong_idx] += 1
        val_prob_metrics[val_idx, idx] = val_prob
        val_pred_metrics[val_idx, idx] = val_pred
        cm = confusion_matrix(y_val_fold, val_pred)
        print(f"Number of False Positives: {cm[0][1]}, Number of False Negatives: {cm[1][0]} in Validation set")


        # test prediction
        test_prob = model.predict(X_test_pt, num_iteration=model.best_iteration)
        test_pred = (test_prob >= best_threshold).astype(int)
        # known test data
        known_test_cm = confusion_matrix(y_test_known, test_pred[concat_test_idx])
        print(f"Number of False Positives: {known_test_cm[0][1]}, Number of False Negatives: {known_test_cm[1][0]} in Known test set")

        pub_pos = np.sum(test_pred[:25108])
        pri_pos = np.sum(test_pred[25108:])
        print(f"Public Positive: {pub_pos}, Private Positive: {pri_pos}")
        test_prob_df[f'pred_{idx+1}_{fold+1}'] = test_prob
        test_pred_df[f'pred_{idx+1}_{fold+1}'] = test_pred

        print()
    # print(f"==========================")


test_pred_df.to_csv(f'./result/test_pred.csv', index=False)
test_prob_df.to_csv(f'./result/test_prob.csv', index=False)

save_dcit = {
    'prob_metrics': val_prob_metrics, #(X_train.shape[0], len(seeds)),
    'pred_metrics': val_pred_metrics, #(X_train.shape[0], len(seeds)),
    'thres_rec': thres_rec,           #(len(seeds), num_splits)
    'f1_rec': f1_rec,                 #(len(seeds), num_splits)
    'prec_rec': prec_rec,             #(len(seeds), num_splits)
    'recall_rec': recall_rec,         #(len(seeds), num_splits)
    'wrong_cnt': val_wrong_cnt,       #(X_train.shape[0], )
}
np.savez_compressed(f'./result/val.npz', **save_dcit)