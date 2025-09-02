Data Shape : (200864, 10214)
Label percentage : 199394(99.27%) vs 1470(0.7318%)
Thre no duplicated ID

group : ['外資券商', '主力券商', '官股券商', '個股券商分點籌碼分析', '個股券商分點區域分析', '個股主力買賣超統計', '日外資', '日自營', '日投信', '技術指標', '月營收', '季IFRS財報', '買超', '賣超', '個股', '上市']

# Missing Ratio Category
10-15%    5144
15-20%    2974
1-10%     1543
0-1%       493
20-40%      44

# Feature Version
* feature v1 : drop 券商代號 (200864, 9584)

    filtered_col = df.filter(like="券商代號", axis=1).columns
    df = df.drop(columns=filtered_col)

* feature v2 : using lgbm select best 500 feature to retrain model

* feature v3 : using full features to train lgbm, and select top 1000 features by **abs_sum(影響力大小)**
train file : fis    result file : feature_select/

* feature v4 : using lambda l1 to select feature in original data

* feature 3k : using top-3000 of feature **feature v3**, and retrain lgbm. Select most counts which in top-K within 15 seeds




# Result

* Template
Val - F1: , Precision: , Recall:  ()
Tes - F1: , Precision: , Recall: 

*  163 braces:
Val - F1: 0.7911, Precision: 0.8664, Recall: 0.7279
Tes - F1: 0.7492, Precision: 0.7791, Recall: 0.7216

* ✅ submission_lgbm_v3_14.csv
Tes - F1: 0.7586, Precision: 0.9649, Recall: 0.625 (110/114)

* lgg_0.902_0.738_144
Tes - F1: 0.9028, Precision: 0.9028, Recall: 0.7386 (130/144)

---
* ✅ fv4_t500_c5
Val - F1: 0.8109, Precision: 0.9498, Recall: 0.7075 
Tes - F1: 0.8000, Precision: 0.9065, Recall: 0.7159 (126/139)

* fv4_419_cb 
Val - F1: 0.8029, Precision: 0.8755, Recall: 0.7415 
Tes - F1: 0.7928, Precision: 0.8408, Recall: 0.7500 (157)

* ✅ fv4_t500_c5_op22 
Val - F1: 0.8152, Precision: 0.9264, Recall: 0.7279
Tes - F1: 0.7939, Precision: 0.8658, Recall: 0.7330 (149)

* fv4_419_0.827_0.782_185
Val - F1: 0.8038, Precision: 0.827 , Recall: 0.782
Tes - F1: 0.7590, Precision: 0.7405, Recall: 0.7784 (137/185)


---
* fv4_105
Val - F1: 0.8180, Precision: 0.9121, Recall: 0.7415 (149)
Val - F1: 0.7965, Precision: 0.8303, Recall: 0.7653 (162)
Tes - F1: , Precision: , Recall: 

* ✅ fv4_105_0.884_0.779_159
Val - F1: 0.8282, Precision: 0.884 , Recall: 0.779
Tes - F1: 0.8119, Precision: 0.8553, Recall: 0.7727 (135/159)

* fv4_105_sclar_pse_0.9218_0.7619_148.txt.csv
Val - F1: 0.8343, Precision: 0.9218, Recall: 0.7619
Tes - F1: 0.8272, Precision: 0.9054, Recall: 0.7614 (134/148)


---
* com_v1
Val - F1: 0.8080, Precision: 0.8735, Recall: 0.7517
Tes - F1: 0.7807, Precision: 0.828, Recall: 0.7386 (129/157)

* ✅ com_v1_opt
Val - F1: 0.8220, Precision: 0.8924, Recall: 0.7619
Tes - F1: 0.7904, Precision: 0.8354, Recall: 0.7500 (132/158)




* ✅ enemble_v1_0.66(2/3)
['com_v1_opt.csv', 'feat_v4_t500_c5_op22.csv', 'feat_v4_t500_c5.csv']
Tes - F1: 0.8013, Precision: 0.9007, Recall: 0.7216 (127/141)
---------------
Prec 9

*  prc9_smt.csv
Tes - F1: 0.8274, Precision: 0.9695, Recall: 0.7216 (127/131)

* ✅ submission_lgbm_v3_14.csv
Tes - F1: 0.7586, Precision: 0.9649, Recall: 0.625 (110/114)

* 200F_Full_139_144
Tes - F1: 0.8317, Precision: 0.9424, Recall: 0.7443 (131/144)

* ✅ fv4_t500_c5
Val - F1: 0.8109, Precision: 0.9498, Recall: 0.7075 
Tes - F1: 0.8000, Precision: 0.9065, Recall: 0.7159 (126/139)

* fv4_105_sclar_pse_0.9218_0.7619_148.txt.csv
Val - F1: 0.8343, Precision: 0.9218, Recall: 0.7619
Tes - F1: 0.8272, Precision: 0.9054, Recall: 0.7614 (134/148)

* lgg_0.902_0.738_144
Tes - F1: 0.9028, Precision: 0.9028, Recall: 0.7386 (130/144)

* ✅ enemble_v1_0.66(2/3)
['com_v1_opt.csv', 'feat_v4_t500_c5_op22.csv', 'feat_v4_t500_c5.csv']
Tes - F1: 0.8013, Precision: 0.9007, Recall: 0.7216 (127/141)


* 200F_138_142
precision 0.9281




--- ONE_ZERO
* prc9_smt_127 (125/127) imp_zero:[780, 20029]

* prc9_smt (127/131)

---
* one134 (130/134)  one:[15012, ], zero:[8806, 13181, 6833]

* one130 (129/130) one: [16292, ]

* one129 (128/129) uncertain:[2195, 14120]

* one127 (127/127)

* 




--- 
ONE  : [16292, 15012, 17378, 10542]
ZERO : [13181, 8806, 6833, 780, 20029, 13338, 755]