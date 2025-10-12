import pandas as pd
import numpy as np
import sys
from collections import defaultdict
data = pd.read_csv('../data/raw/data_raw')


#check for missing values
def delete_nulls(data):
    for column in data.columns:
        print(data[column].isnull().sum())
#checking variables for ordinal meaning
def correlation_test(column_name,data,output=False):
    variables = {}
    for variable in data[column_name].unique():
        classes,counts = np.unique(data[data[column_name]==variable]['class'],return_counts=True)
        ratio = 'No values'
        #print(counts)
        if len(counts) == 2:
            ratio = [counts[0]/(counts[0] + counts[1]), counts[1]/(counts[0]+counts[1])]
            ratio[0],ratio[1] = round(float(ratio[0]),1),round(float(ratio[1]),1)
            if ratio[0] > ratio[1]:
                ratio.append('EDIBLE')
            else:
                ratio.append('POISONOUS')

        elif len(counts) == 1:
            if classes == 'EDIBLE':
                ratio = [1,0,'EDIBLE']
            else:
                ratio = [0,1,'POISONOUS']
        if output:
            if ratio[0] > ratio[1]:
                print(variable,ratio)
            elif ratio[0] < ratio[1]:
                print(variable,ratio)
            else:
                print(variable,'EVEN')
        #something wrong with this, figure it out!!
        variables[variable] = ratio
    return variables        
for column in data.columns:
    #print(column)
    correlation_test(column,data)
def check_for_wrong_data(data):
    for column in data:
        print(column,data[column].unique())
def ordinality_check(column_name,size,volume):
    #set up base variables
    iters = np.random.randint(volume,size=size)
    past_state = 0
    consistency = defaultdict(list)
    consistent = []
    #run each iteration
    for i in iters:
        #select random sample
        sample = data.sample(500,random_state=i)
        #get variables into a dict
        vars = correlation_test(column_name,sample)
        #get a system that checks for consistency
        for var in vars:
            consistency[var].append(vars[var][2])
    for var in consistency:
        flag = 0
        for i in range(len(consistency[var])-1):
            if consistency[var][i] != consistency[var][i+1]:
                flag = 1
        if flag == 0:
            consistent.append(var)
    if len(consistent) == len(consistency):
        print(column_name,'CONSISTENT')
        return column_name

def analyse_ordinality(columns,size,volume):
    ordinal = []
    for column in columns:
        print(f'----------Check consistency for {column}----------')
        verdict = ordinality_check(column,size,volume)
        if verdict != None:
            ordinal.append(verdict)
        print(f'-----------End of check for {column}----------')
    print('ORDINAL COLUMNS: ', ordinal)
    return ordinal
#print(data[data['cap-shape']=='nan'])
#print(check_for_wrong_data(data))
#print(data.columns)

'''List of the variables that appear to have ordinal meaning
-gill-size
-ring-number
-cap-shape
-cap-surface
-stalk-surface-below-ring
'''
possibly_ordinal = ['gill-size','ring-number','cap-shape','cap-surface','stalk-surface-below-ring']
analyse_ordinality(data.columns,1000,100)


















