import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve


NUM = 3000
CUM = 10
folder = 'full_feat'

print("Loading data...")
df = pd.read_parquet("./Data/training_feature_v1.parquet")

# ========== 讀取特徵重要性 ==========
gain = pd.read_csv(filepath_or_buffer=f'./{folder}/fi_gain.csv', index_col=0)
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
print("X shape:", X.shape)

# assert False

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, 
                                                      stratify=y, random_state=42)

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

# 5. 設定參數
params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'data_sample_strategy': 'goss',
    'metric': ['binary_logloss', 'f1', 'precision', 'recall'],
    # 'metric': ['binary_logloss'],
    'is_unbalance': True,
    'verbosity': -1,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,

    'pos_subsample': 1.0,
    'neg_subsample': 0.02,
    'feature_fraction': 0.8,
    # 'bagging_fraction': 0.8,
    # 'bagging_freq': 5,
    
    'device': 'gpu',
    'n_jobs': 70,
}


print("Start training...")
# 6. 訓練 model
model = lgb.train(
    params,
    train_set=train_set,
    valid_sets=[valid_set],
    valid_names=['valid'],
    num_boost_round=2000,
    feval=[f1_eval, precision_eval, recall_eval],
    callbacks=[
        lgb.early_stopping(stopping_rounds=200, first_metric_only=True, verbose=True),
        lgb.log_evaluation(period=50),
        ],
)

# 儲存模型
model.save_model(f'./{folder}/fullfeat_top{NUM}-cum{CUM}.txt')
    
y_pred_prob = model.predict(X_valid, num_iteration=model.best_iteration)

# 10. 加上最佳 threshold (最大 F1)
prec, recall, thresholds = precision_recall_curve(y_valid, y_pred_prob)
f1s = 2 * prec * recall / (prec + recall + 1e-10)
best_idx = np.argmax(f1s)
best_threshold = thresholds[best_idx]
print(f"\nBest threshold for max F1: {best_threshold:.4f}, F1: {f1s[best_idx]:.4f}, Precision: {prec[best_idx]:.4f}, Recall: {recall[best_idx]:.4f}")