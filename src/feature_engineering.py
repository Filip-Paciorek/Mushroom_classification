import pandas as pd
import numpy as np
from exploratory_data_analysis import get_ordinal_results
data_raw = pd.read_csv('../data/raw/data_raw')
data_raw = data_raw[data_raw['stalk-root']!='?'].copy()
#data_raw.drop(columns=['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.2'],inplace=True)
#data_raw.to_csv('../data/raw/data_raw',index=False)
results = get_ordinal_results()
print(results)
def base_model_features(data_raw):
    data_clean = data_raw
    for column in data_raw:
        if column == 'class':
            data_raw[data_raw['class']=='EDIBLE'] = 1
            data_raw[data_raw['class']=='POISONOUS']=0
            continue
        data_clean = pd.get_dummies(data_clean,columns=[column])
    return data_clean
def ordinal_max_features(data_raw):
    features_with_ordinal_meaning = ['gill-size','ring-number','cap-shape','cap-surface','stalk-surface-below-ring']
data_clean = base_model_features(data_raw)
data_clean.to_csv('../data/processed/base_model_data',index=False)
#print(data_raw['stalk-root'].nunique())
