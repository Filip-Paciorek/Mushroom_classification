import pandas as pd
import numpy as np
from exploratory_data_analysis import get_ordinal_results
data_raw = pd.read_csv('../data/raw/data_raw')
data_raw = data_raw[data_raw['stalk-root']!='?'].copy()
#data_raw.drop(columns=['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.2'],inplace=True)
#data_raw.to_csv('../data/raw/data_raw',index=False)
results = get_ordinal_results()
def no_obvious_features(state):
    if state == 'mild':
        return ['odor','gill-color','spore-print-color']
    if state == 'severe':
        return ['odor','gill-color','gill-size','bruises','spore-print-color']
    return []
def base_model_features(data_raw,state=None):
    data_raw = data_raw.drop(no_obvious_features(state),axis=1)
    data_clean = data_raw
    for column in data_raw:
        if column == 'class':
            data_clean.loc[data_raw['class']=='EDIBLE','class'] = 1
            data_clean.loc[data_raw['class']=='POISONOUS','class']=0
            continue
        data_clean = pd.get_dummies(data_clean,columns=[column])
    return data_clean
def ordinal_max_features(data_raw,ord_columns,state=None):
    rejected_columns = no_obvious_features(state)
    data_raw = data_raw.drop(rejected_columns,axis=1)
    data_clean = data_raw
    for column in ord_columns:
        if column in rejected_columns:
            continue
        safety_score = len(ord_columns[column])
        if column == 'class':
            data_clean.loc[data_clean[column] == 'EDIBLE',column] = 1
            data_clean.loc[data_clean[column] == 'POISONOUS',column] = 0
        for group in ord_columns[column]:
            if len(group) == 1:
                data_clean.loc[data_clean[column] == group[0],column] = safety_score
            else:
                for characteristic in group:
                    #print(characteristic)
                    data_clean.loc[data_clean[column] == characteristic,column] = safety_score
            safety_score -= 1
    for column in data_clean:
        if column not in ord_columns:
            data_clean = pd.get_dummies(data_clean,columns=[column])
    return data_clean

data_clean = ordinal_max_features(data_raw,results)
data_clean.to_csv('../data/processed/max_ordinal_data',index=False)
data_clean = ordinal_max_features(data_raw,results,'mild')
data_clean.to_csv('../data/processed/no_obvious_data_ordinal_mild',index=False)
data_clean = ordinal_max_features(data_raw,results,'severe')
data_clean.to_csv('../data/processed/no_obvious_data_ordinal_severe',index=False)
data_clean = base_model_features(data_raw,'mild')
data_clean.to_csv('../data/processed/no_obvious_data_base_mild',index=False)    
data_clean = base_model_features(data_raw,'severe')
data_clean.to_csv('../data/processed/no_obvious_data_base_severe',index=False)    
data_clean = base_model_features(data_raw)
data_clean.to_csv('../data/processed/base_model_data',index=False)
#print(data_raw['stalk-root'].nunique())
