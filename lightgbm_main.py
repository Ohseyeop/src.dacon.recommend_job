from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from pyarrow import csv
from xgboost import XGBClassifier

r_opt = csv.ReadOptions(use_threads=True)
train = csv.read_csv('./data/train.v5.csv', read_options=r_opt).to_pandas()
valid = csv.read_csv('./data/valid.v5.csv', read_options=r_opt).to_pandas()
test = csv.read_csv('./data/test.v5.csv', read_options=r_opt).to_pandas()
output = open("./result.log", 'w')
# train = train.drop(['contents_attribute_j_2','person_attribute_a_2','contents_attribute_k_2'
#                   ,'person_prefer_c_4','contents_attribute_c_4'],axis=1)
# valid = valid.drop(['contents_attribute_j_2','person_attribute_a_2','contents_attribute_k_2'
#                   ,'person_prefer_c_4','contents_attribute_c_4'],axis=1)
# test = test.drop(['contents_attribute_j_2','person_attribute_a_2','contents_attribute_k_2'
#                   ,'person_prefer_c_4','contents_attribute_c_4'],axis=1)

train_x = train.values[:,1:-1]
train_y = train.values[:,-1:]

valid_x = valid.values[:,1:-1]
valid_y = valid.values[:,-1:]

test_x = test.values[:,1:]
test_id = test.values[:,0:1]


# evals = [(valid_x, valid_y)]

# lgbm = LGBMClassifier(max_depth=10, min_child_samples=20, subsample=0.8)
# lgbm.fit(train_x, train_y, early_stopping_rounds=100, eval_metric='auc', eval_set=[(valid_x, valid_y)], verbose=True)
# best_model = lgbm

# lgbm = LGBMClassifier()
# params = {
#           'n_estimators' : [5000],
#           'learning_rate': [0.05,0.01],
#           'max_depth': [30, 20],
#           'min_child_samples': [100, 70],
#           'subsample': [0.95]}

# grid = GridSearchCV(lgbm, param_grid=params)
# grid.fit(train_x, train_y.ravel(), early_stopping_rounds=100, eval_metric='auc',
#          eval_set=[(valid_x, valid_y.ravel())])


#grid = XGBClassifier(use_label_encoder=False,tree_method = 'hist',single_precision_histogram=True,
#        learning_rate=0.1,max_depth=10,min_split_loss=0.2,min_child_weight=650,n_estimators=5000,reg_alpha=0.75,reg_lambda=0.85,sub_sample=0.9,colsample_bytree=0.9)
#grid.fit(train_x, train_y.ravel(), early_stopping_rounds=100, eval_metric='error',eval_set=[(valid_x, valid_y.ravel())])
xgb=XGBClassifier(use_label_encoder=False,tree_method = 'hist',single_precision_histogram=True)
# params={'booster' :['gbtree'],
#                  'max_depth':[28],
#                  'min_child_weight':[600],
#                  'n_estimators':[5000],
#                  'objective':['binary:logistic'],
#                  'random_state':[2],
#                  'reg_lambda':[0.99],
#                  'reg_alpha':[0.9]} 

params={'booster' :['gbtree'],
                 'learning_rate':[0.1],
                 'min_split_loss':[0.5,1,5],
                 'max_depth':[34,35,36],
                 'min_child_weight':[550],
                 'n_estimators':[5000],
                 'objective':['binary:logistic'],
                 'random_state':[2],
                 'colsample_bytree':[0.9,0.95,0.8],
                 'reg_lambda':[0.85],
                 'reg_alpha':[0.75]} 

# booster (gbtree , gblinear)
# nthread 스레트 cpu
# learning_rate
# n_estimators
# min_split_loss/gamma 리프노드의 추가분할을 결정할 최소손실 감소값 0 ~ unlimit
# colsample_bytree 트리 생성에 필요한 피처의 샘플링에 사용 피처가 많을 때 과적합 조절에 사용 0 ~ 1
# lambda/reg_lambda L2
# alpha/reg_alpha  L1
grid = GridSearchCV(xgb, param_grid=params)
grid.fit(train_x, train_y.ravel(), early_stopping_rounds=100, eval_metric='error',
         eval_set=[(valid_x, valid_y.ravel())])

output.write('best parameters : '+ str(grid.best_params_)+"\n")
output.write('best score : '+ str(grid.best_score_)+"\n")

print('best parameters : ', grid.best_params_)
print('best score : ', grid.best_score_)
best_model = grid.best_estimator_


preds =best_model.predict(valid_x)
pred_proba = best_model.predict_proba(valid_x)[:,1]

preds_sub = best_model.predict(test_x)
submission = np.concatenate((test_id, preds_sub.reshape(-1,1)), axis=1)
submission = pd.DataFrame(submission,columns=['id','target'])

submission.to_csv('./data/result.csv', index=False)


from sklearn.metrics import *

confusion = confusion_matrix(valid_y, preds) 
accuracy = accuracy_score(valid_y, preds) 
precision = precision_score(valid_y, preds) 
recall = recall_score(valid_y, preds) 
f1 = f1_score(valid_y, preds) 
# ROC AUC 
roc_auc = roc_auc_score(valid_y, pred_proba) 
output.write('Confusion Matrix'+"\n")
output.write(str(confusion)+"\n") 
output.write('accuracy: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1: {3:.4f}, roc_auc: {4:.4f}'.format( accuracy, precision, recall, f1, roc_auc)+"\n")
print('Confusion Matrix') 
print(confusion) 
print('accuracy: {0:.4f}, precision: {1:.4f}, recall: {2:.4f}, f1: {3:.4f}, roc_auc: {4:.4f}'.format( accuracy, precision, recall, f1, roc_auc))
    

# get_clf_eval()를 이용해 사키릿런 래퍼 XGBoost로 만들어진 모델 예측 성능 평가
# get_clf_eval(valid_y, preds, pred_proba,output)


output.close()
# from xgboost import plot_importance
# # from lightgbm import plot_importance
# import matplotlib.pyplot as plt
# import matplotlib.image as img

# fig, ax = plt.subplots(figsize=(10, 12))
# plot_importance(best_model, ax=ax)
# plt.savefig(f'./param.jpg',dpi=300)
