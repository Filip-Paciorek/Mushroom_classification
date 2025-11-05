import pandas as pd
import numpy as np
import sys
import os
import joblib
from collections import defaultdict
#check for missing values
def delete_nulls(data):
    for column in data.columns:
        print(data[column].isnull().sum())
#checking variables for ordinal meaning
def correlation_test(column_name,data,output=False):
    variables = {}
    #go through each variable in the column
    for variable in data[column_name].unique():
        #look for how much of the variable is poisonous and how much is edible
        classes,counts = np.unique(data[data[column_name]==variable]['class'],return_counts=True)
        ratio = 'No values'
        #print(counts)
        #calculate ratio
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
        ratio.append(variable)
        #purely visual part of code
        if output:
            if ratio[0] > ratio[1]:
                print(variable,ratio)
            elif ratio[0] < ratio[1]:
                print(variable,ratio)
            else:
                print(variable,'EVEN')
        #add the ratio to the dict
        variables[variable] = ratio
    return variables
#sorting function
def sort_by_edibility(characteristics):
    ranked_characteristics = {}
    sorted_characteristics = [0]*len(characteristics)
    #for each item
    for item in characteristics:
        #add a counter to each variable
        ranked_characteristics[item] = [item,0]
        for item2 in characteristics:
            if characteristics[item] < characteristics[item2]:
                ranked_characteristics[item][1] += 1
        #if its the first occurence
        if sorted_characteristics[ranked_characteristics[item][1]] == 0:
            sorted_characteristics[ranked_characteristics[item][1]] = [ranked_characteristics[item][0]]
        #if its a repeated occurence
        else:
            sorted_characteristics[ranked_characteristics[item][1]].append(ranked_characteristics[item][0])
    #get rid of 0s in ordered list
    sorted_characteristics = [x for x in sorted_characteristics if x != 0] 

    return sorted_characteristics

def check_for_wrong_data(data):
    for column in data:
        print(column,data[column].unique())
def ordinality_check(column_name,data,size,volume):
    #set up base variables
    iters = np.random.randint(volume,size=size)
    past_state = 0
    consistency = defaultdict(list)
    consistent = []
    mean_edibility = defaultdict(float)
    #run each iteration
    for i in iters:
        #select random sample
        sample = data.sample(500,random_state=i)
        #get variables into a dict
        vars = correlation_test(column_name,sample)
        #get a system that checks for consistency
        for var in vars:
            #cumulate the % of edibility and add it to the dict entry of certain characteristic
            mean_edibility[vars[var][3]] += vars[var][0]
            #consistency[var].append(vars[var][3])
            #append whethere characteristic is mostly edible or poisonous in this iteration
            consistency[var].append(vars[var][2])
    #make it truly a mean
    for item in mean_edibility:
        mean_edibility[item] = mean_edibility[item]/size
    #sort
    ordered_characteristics = sort_by_edibility(mean_edibility)
    for var in consistency:
        #look for any difference between major edibility/inedibility, if it appears, flag as inedible
        flag = 0
        for i in range(len(consistency[var])-1):
            if consistency[var][i] != consistency[var][i+1]:
                flag = 1
        if flag == 0:
            consistent.append(var)
    if len(consistent) == len(consistency):
        print(column_name,'CONSISTENT')
        return column_name,ordered_characteristics

def analyse_ordinality(columns,data,size,volume,include_characteristics=False):
    ordinal = {}
    for column in columns:
        print(f'----------Check consistency for {column}----------')
        verdict = ordinality_check(column,data,size,volume)
        if verdict != None:
            ordinal[verdict[0]] = verdict[1]
        if include_characteristics == True and verdict != None:
            print(f'{column}: {verdict[1]}')
        print(f'-----------End of check for {column}----------')
    if include_characteristics == False:
        print('ORDINAL COLUMNS: ', ordinal.keys())
    return ordinal
def get_ordinal_results(data_path='../data/raw/data_raw',cache_path='../data/interim/ordinal_results.plk',force_compute = False):
    if os.path.exists(cache_path) and not force_compute:
        print(f'Loading ordinal variables from: {cache_path}')
        return joblib.load(cache_path)
    print('Running full EDA analysis')
    data = pd.read_csv(data_path)
    print(data)
    ordinal_columns = analyse_ordinality(data.columns,data,size=100, volume=100,include_characteristics=True)
    results = ordinal_columns
    joblib.dump(results, cache_path)
    print('EDA analysis finished')
    return results

if __name__ == '__main__':
    results = get_ordinal_results(force_compute=True)

















