import pandas as pd
import numpy as np

data_raw = pd.read_csv('../data/raw/data_raw')
data_raw = data_raw[data_raw['stalk-root']!='?'].copy()
#data_raw.drop(columns=['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.2'],inplace=True)
#data_raw.to_csv('../data/raw/data_raw',index=False)
def base_model_features(data_raw):
    data_clean = data_raw
    for column in data_raw:
        data_clean = pd.get_dummies(data_clean,columns=[column])
    return data_clean
data_clean = base_model_features(data_raw)
data_clean.to_csv('../data/processed/base_model_data',index=False)
#print(data_raw['stalk-root'].nunique())
