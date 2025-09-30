import pandas as pd
import numpy as np


data = pd.read_csv('../data/raw/data_raw')


#check for missing values
def delete_nulls(data):
    for column in data.columns:
        print(data[column].isnull().sum())
#checking variables with ordinal meaning
def correlation_test(column_name):
    for variable in data[column_name].unique():
        classes,counts = np.unique(data[data[column_name]==variable]['class'],return_counts=True)
        ratio = 'No values'
        if len(counts) == 2:
            ratio = [counts[0]/(counts[0] + counts[1]), counts[1]/(counts[0]+counts[1])]
            ratio[0],ratio[1] = round(float(ratio[0]),1),round(float(ratio[1]),1)
        elif len(counts) == 1:
            print(classes)
            ratio = 1
        print(ratio)        
for column in data.columns:
    print(column)
    correlation_test(column)
def check_for_wrong_data(data):
    for column in data:
        print(column,data[column].unique())
print(data[data['cap-shape']=='nan'])
#print(check_for_wrong_data(data))
print(data.columns)
'''List of the variables that appear to have ordinal meaning
-gill-size
-ring-number
'''

