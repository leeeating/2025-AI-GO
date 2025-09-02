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
    'max_depth': -1,
    # 'min_data_in_leaf': 500,

    'feature_fraction': 0.7,
    'bagging_fraction': 0.75,
    # 'top_rate': 0.25,
    # 'other_rate': 0.5,

    'n_jobs': 30,
}

----------

## Using **ALL Data** to PowerTransformer scaler

* Augment Valid set
valid's f1: 0.832512	valid's precision: 0.884817	valid's recall: 0.786047

* Valid set
F1: 0.8096, Precision: 0.8866, Recall: 0.7449
Public test num pos: 150, Private test num pos: 141
False positive: 16, False negative: 17, Wrong: 33


## Using **Train Data** to PowerTransformer scaler
* Augment Valid set
valid's f1: 0.830694	valid's precision: 0.872123	valid's recall: 0.793023

* Valid set
F1: 0.8008, Precision: 0.9531, Recall: 0.6905
Public test num pos: 131, Private test num pos: 125
False positive: 16, False negative: 26, Wrong: 42

----------

=================================

----------

## Using **ALL Data** to Standard scaler

* Augment Valid set
valid's f1: 0.825029	valid's precision: 0.822171	valid's recall: 0.827907

* Valid set
F1: 0.7985, Precision: 0.8945, Recall: 0.7211
Public test num pos: 144, Private test num pos: 141
False positive: 16, False negative: 19, Wrong: 35



## Using **Train Data** to Standard scaler

* Augment Valid set
valid's f1: 0.827417	valid's precision: 0.873385	valid's recall: 0.786047

* Valid set
F1: 0.7963, Precision: 0.8740, Recall: 0.7313
Public test num pos: 144, Private test num pos: 139
False positive: 16, False negative: 18, Wrong: 34



# 153
'num_leaves': 128,
'min': 20
    Best threshold: 0.0031, F1: 0.8037, Precision: 0.8821, Recall: 0.7381
    Validation num pos: 246

    False positive: 16, False negative: 15, Wrong: 31
    Before retain,  Public test num pos: 152, Private test num pos: 147


'num_leaves': 64,
'min': 20
    Best threshold: 0.0587, F1: 0.7992, Precision: 0.9755, Recall: 0.6769
    Validation num pos: 204

    False positive: 15, False negative: 31, Wrong: 46
    Before retain,  Public test num pos: 124, Private test num pos: 122



# 105
'num_leaves': 128,
'min': 20
    Best threshold: 0.0038, F1: 0.7913, Precision: 0.8482, Recall: 0.7415
    Validation num pos: 257

    False positive: 17, False negative: 13, Wrong: 30
    Before retain,  Public test num pos: 143, Private test num pos: 137



'num_leaves': 64,
'min': 20
    Best threshold: 0.0101, F1: 0.7909, Precision: 0.8966, Recall: 0.7075
    Validation num pos: 232

    False positive: 16, False negative: 18, Wrong: 34
    Before retain,  Public test num pos: 143, Private test num pos: 137


## weight test
* 'is_unbalance': True,
Best threshold: 0.0035, F1: 0.8315, Precision: 0.8788, Recall: 0.7891
Validation num pos: 132
Before retain,  Public test num pos: 149, Private test num pos: 155


* 'scale_pos_weight': rate
Best threshold: 0.0392, F1: 0.8189, Precision: 0.9720, Recall: 0.7075
Validation num pos: 107
Before retain,  Public test num pos: 124, Private test num pos: 125


* 'scale_pos_weight': 120
Best threshold: 0.0042, F1: 0.8218, Precision: 0.8828, Recall: 0.7687
Validation num pos: 128
Before retain,  Public test num pos: 138, Private test num pos: 144


* 'scale_pos_weight': 135
Best threshold: 0.0256, F1: 0.8263, Precision: 0.9554, Recall: 0.7279
Validation num pos: 112
Before retain,  Public test num pos: 128, Private test num pos: 132

* 'scale_pos_weight': 150
Best threshold: 0.0020, F1: 0.8214, Precision: 0.8647, Recall: 0.7823
Validation num pos: 133
Before retain,  Public test num pos: 147, Private test num pos: 158

* 'scale_pos_weight': 200
Best threshold: 0.0111, F1: 0.8195, Precision: 0.9160, Recall: 0.7415
Validation num pos: 119
Before retain,  Public test num pos: 132, Private test num pos: 136


* 'scale_pos_weight': 20
Best threshold: 0.0037, F1: 0.8266, Precision: 0.9032, Recall: 0.7619
Validation num pos: 124
Before retain,  Public test num pos: 139, Private test num pos: 145

* weight=rate(pos)
Best threshold: 0.0026, F1: 0.8143, Precision: 0.8571, Recall: 0.7755
Validation num pos: 133
Before retain,  Public test num pos: 148, Private test num pos: 154

* weight=rate(pos + hard neg)
Best threshold: 0.0017, F1: 0.8198, Precision: 0.8529, Recall: 0.7891
Validation num pos: 136
Before retain,  Public test num pos: 151, Private test num pos: 155

* weight=rate(pos + hard neg*2)
Best threshold: 0.0527, F1: 0.8127, Precision: 0.9808, Recall: 0.6939
Validation num pos: 104
Before retain,  Public test num pos: 124, Private test num pos: 123

* weight=rate(pos + hard neg*0.5)
Best threshold: 0.0011, F1: 0.8097, Precision: 0.8239, Recall: 0.7959
Validation num pos: 142
Before retain,  Public test num pos: 161, Private test num pos: 173