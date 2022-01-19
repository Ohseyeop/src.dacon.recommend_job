from numpy.lib.function_base import kaiser
import pandas as pd
import torch
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from typing import Dict

def add_code(df: pd.DataFrame,
    d_code: Dict[int, Dict[str, int]], 
    h_code: Dict[int, Dict[str, int]], 
    l_code: Dict[int, Dict[str, int]]):
    
    # Copy input data
    df = df.copy()   

    # D Code
    df['person_prefer_d_1_n'] = df['person_prefer_d_1'].apply(lambda x: d_code[x]['속성 D 세분류코드'])
    df['person_prefer_d_1_s'] = df['person_prefer_d_1'].apply(lambda x: d_code[x]['속성 D 소분류코드'])
    df['person_prefer_d_1_m'] = df['person_prefer_d_1'].apply(lambda x: d_code[x]['속성 D 중분류코드'])
    df['person_prefer_d_1_l'] = df['person_prefer_d_1'].apply(lambda x: d_code[x]['속성 D 대분류코드'])

    df['person_prefer_d_2_n'] = df['person_prefer_d_2'].apply(lambda x: d_code[x]['속성 D 세분류코드'])
    df['person_prefer_d_2_s'] = df['person_prefer_d_2'].apply(lambda x: d_code[x]['속성 D 소분류코드'])
    df['person_prefer_d_2_m'] = df['person_prefer_d_2'].apply(lambda x: d_code[x]['속성 D 중분류코드'])
    df['person_prefer_d_2_l'] = df['person_prefer_d_2'].apply(lambda x: d_code[x]['속성 D 대분류코드'])

    df['person_prefer_d_3_n'] = df['person_prefer_d_3'].apply(lambda x: d_code[x]['속성 D 세분류코드'])
    df['person_prefer_d_3_s'] = df['person_prefer_d_3'].apply(lambda x: d_code[x]['속성 D 소분류코드'])
    df['person_prefer_d_3_m'] = df['person_prefer_d_3'].apply(lambda x: d_code[x]['속성 D 중분류코드'])
    df['person_prefer_d_3_l'] = df['person_prefer_d_3'].apply(lambda x: d_code[x]['속성 D 대분류코드'])

    df['contents_attribute_d_n'] = df['contents_attribute_d'].apply(lambda x: d_code[x]['속성 D 세분류코드'])
    df['contents_attribute_d_s'] = df['contents_attribute_d'].apply(lambda x: d_code[x]['속성 D 소분류코드'])
    df['contents_attribute_d_m'] = df['contents_attribute_d'].apply(lambda x: d_code[x]['속성 D 중분류코드'])
    df['contents_attribute_d_l'] = df['contents_attribute_d'].apply(lambda x: d_code[x]['속성 D 대분류코드'])

    # H Code
    df['person_prefer_h_1_l'] = df['person_prefer_h_1'].apply(lambda x: h_code[x]['속성 H 대분류코드'])
    df['person_prefer_h_1_m'] = df['person_prefer_h_1'].apply(lambda x: h_code[x]['속성 H 중분류코드'])
    
    df['person_prefer_h_2_l'] = df['person_prefer_h_2'].apply(lambda x: h_code[x]['속성 H 대분류코드'])
    df['person_prefer_h_2_m'] = df['person_prefer_h_2'].apply(lambda x: h_code[x]['속성 H 중분류코드'])
    
    df['person_prefer_h_3_l'] = df['person_prefer_h_3'].apply(lambda x: h_code[x]['속성 H 대분류코드'])
    df['person_prefer_h_3_m'] = df['person_prefer_h_3'].apply(lambda x: h_code[x]['속성 H 중분류코드'])

    df['contents_attribute_h_l'] = df['contents_attribute_h'].apply(lambda x: h_code[x]['속성 H 대분류코드'])
    df['contents_attribute_h_m'] = df['contents_attribute_h'].apply(lambda x: h_code[x]['속성 H 중분류코드'])

    # L Code
    df['contents_attribute_l_n'] = df['contents_attribute_l'].apply(lambda x: l_code[x]['속성 L 세분류코드'])
    df['contents_attribute_l_s'] = df['contents_attribute_l'].apply(lambda x: l_code[x]['속성 L 소분류코드'])
    df['contents_attribute_l_m'] = df['contents_attribute_l'].apply(lambda x: l_code[x]['속성 L 중분류코드'])
    df['contents_attribute_l_l'] = df['contents_attribute_l'].apply(lambda x: l_code[x]['속성 L 대분류코드'])
    
    return df

# data = pd.read_csv('./data/train.csv' ,',')
# data = data.drop(['d_l_match_yn','d_m_match_yn','d_s_match_yn','h_l_match_yn','h_m_match_yn','h_s_match_yn',
#                   'person_rn','contents_rn','contents_open_dt'],axis=1)

# train, val = train_test_split(data,test_size = 0.1,shuffle=True, stratify=data.target, random_state=34)


# train.to_csv('./data/train.v1.csv',index=False)
# val.to_csv('./data/valid.v1.csv',index=False)
# print()

d_code = pd.read_csv('./data/attribute_D.csv', encoding='utf8', index_col=0).T.to_dict()
h_code = pd.read_csv('./data/attribute_H.csv', encoding='utf8', index_col=0).T.to_dict()
l_code = pd.read_csv('./data/attribute_L.csv', encoding='utf8', index_col=0).T.to_dict()

train_data = pd.read_csv('./data/train.csv' ,',')
test_data = pd.read_csv('./data/test.csv' ,',')

test_data['target'] = 0

train_objs_num = len(train_data)
data = pd.concat(objs=[train_data, test_data], axis=0)

data = data.drop(['person_rn','contents_rn','contents_open_dt'],axis=1)
data = data.rename(columns={"person_attribute_a_1":"person_attribute_a_b",
                   "contents_attribute_j_1":"contents_attribute_j_b"})

data["d_l_match_yn"] = data["d_l_match_yn"].astype(int)
data["d_m_match_yn"] = data["d_m_match_yn"].astype(int)
data["d_s_match_yn"] = data["d_s_match_yn"].astype(int)
data["h_l_match_yn"] = data["h_l_match_yn"].astype(int)
data["h_m_match_yn"] = data["h_m_match_yn"].astype(int)
data["h_s_match_yn"] = data["h_s_match_yn"].astype(int)

data = add_code(data, d_code, h_code, l_code)

data = pd.get_dummies(data,columns=['person_attribute_a'])
data = pd.get_dummies(data,columns=['person_prefer_c'])
data = pd.get_dummies(data,columns=['person_prefer_f'])
data = pd.get_dummies(data,columns=['person_prefer_g'])
data = pd.get_dummies(data,columns=['contents_attribute_i'])
data = pd.get_dummies(data,columns=['contents_attribute_a'])
data = pd.get_dummies(data,columns=['contents_attribute_j'])
data = pd.get_dummies(data,columns=['contents_attribute_c'])
data = pd.get_dummies(data,columns=['contents_attribute_k'])
data = pd.get_dummies(data,columns=['contents_attribute_m'])

data = pd.get_dummies(data,columns=['person_prefer_d_1_m'])
data = pd.get_dummies(data,columns=['person_prefer_d_1_l'])

data = pd.get_dummies(data,columns=['person_prefer_d_2_m'])
data = pd.get_dummies(data,columns=['person_prefer_d_2_l'])

data = pd.get_dummies(data,columns=['person_prefer_d_3_m'])
data = pd.get_dummies(data,columns=['person_prefer_d_3_l'])

data = pd.get_dummies(data,columns=['contents_attribute_d_m'])
data = pd.get_dummies(data,columns=['contents_attribute_d_l'])


data = pd.get_dummies(data,columns=['person_prefer_h_1_l'])
data = pd.get_dummies(data,columns=['person_prefer_h_1_m'])

data = pd.get_dummies(data,columns=['person_prefer_h_2_l'])
data = pd.get_dummies(data,columns=['person_prefer_h_2_m'])

data = pd.get_dummies(data,columns=['person_prefer_h_3_l'])
data = pd.get_dummies(data,columns=['person_prefer_h_3_m'])

data = pd.get_dummies(data,columns=['contents_attribute_h_l'])
data = pd.get_dummies(data,columns=['contents_attribute_h_m'])

data = pd.get_dummies(data,columns=['contents_attribute_l_m'])
data = pd.get_dummies(data,columns=['contents_attribute_l_l'])

cols = [col for col in data.columns if col != 'target']+['target']

data = data[cols]

print(data.head().columns)

train_preprocessed = data[:train_objs_num]
test_preprocessed = data[train_objs_num:]

train, val = train_test_split(train_preprocessed,test_size = 0.03, shuffle=True, stratify=train_preprocessed.target, random_state=34)

train.to_csv('./data/train.v6.csv',index=False)
val.to_csv('./data/valid.v6.csv',index=False)


test_preprocessed = test_preprocessed.drop(['target'],axis=1)
print(test_preprocessed.head().columns)
test_preprocessed.to_csv('./data/test.v6.csv',index=False)