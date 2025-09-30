import pandas as pd
import numpy as np


data = pd.read_csv('../data/raw/mushroom/expanded',skiprows=8,sep=',',header=0,names=
['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
 'stalk-surface-below-ring', 'stalk-color-above-ring',
 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
 'ring-type', 'spore-print-color', 'population', 'habitat'])
#data.drop(columns=['Unnamed: 0'],inplace=True)
data = data.iloc[:-1].copy()
data.to_csv('../data/raw/data_raw',index=False)
print(data.shape)
