import numpy as np
from sklearn.preprocessing import StandardScaler 
import pandas as pd
import copy
def divide_age(dataset) : 
    dataset['DAYS_BIRTH'] = (-dataset['DAYS_BIRTH'])//365
    dataset.loc[(dataset['DAYS_BIRTH'] < 25) & (20 <= dataset['DAYS_BIRTH']), 'DAYS_BIRTH'] = 0 
    dataset.loc[(dataset['DAYS_BIRTH'] < 30) & (25 <= dataset['DAYS_BIRTH']), 'DAYS_BIRTH'] = 1 
    dataset.loc[(dataset['DAYS_BIRTH'] < 35) & (30 <= dataset['DAYS_BIRTH']), 'DAYS_BIRTH'] = 2 
    dataset.loc[(dataset['DAYS_BIRTH'] < 40) & (35 <= dataset['DAYS_BIRTH']), 'DAYS_BIRTH'] = 3 
    dataset.loc[(dataset['DAYS_BIRTH'] < 45) & (40 <= dataset['DAYS_BIRTH']), 'DAYS_BIRTH'] = 4 
    dataset.loc[(dataset['DAYS_BIRTH'] < 50) & (45 <= dataset['DAYS_BIRTH']), 'DAYS_BIRTH'] = 5 
    dataset.loc[(dataset['DAYS_BIRTH'] < 55) & (50 <= dataset['DAYS_BIRTH']), 'DAYS_BIRTH'] = 6 
    dataset.loc[(dataset['DAYS_BIRTH'] < 60) & (55 <= dataset['DAYS_BIRTH']), 'DAYS_BIRTH'] = 7 
    dataset.loc[(dataset['DAYS_BIRTH'] < 65) & (60 <= dataset['DAYS_BIRTH']), 'DAYS_BIRTH'] = 8 
    dataset.loc[(dataset['DAYS_BIRTH'] < 70) & (65 <= dataset['DAYS_BIRTH']), 'DAYS_BIRTH'] = 9 
    return dataset

def divide_employed(dataset) : 
    dataset.loc[dataset['DAYS_EMPLOYED'] < 0, 'DAYS_EMPLOYED'] = -dataset.loc[dataset['DAYS_EMPLOYED'] < 0, 'DAYS_EMPLOYED']
    dataset.loc[dataset['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = 0
    x = copy.deepcopy(dataset)
    lower = np.percentile(x['DAYS_EMPLOYED'], 0)  
    for p in range(10, 110, 10) : 
        upper = np.percentile(x['DAYS_EMPLOYED'], p) 
        dataset.loc[((lower <= dataset['DAYS_EMPLOYED']) & (dataset['DAYS_EMPLOYED'] <= upper)), 'DAYS_EMPLOYED'] = p // 10
    return dataset

def divide_begin_month(dataset) : 
    x = copy.deepcopy(dataset)
    lower = np.percentile(x['begin_month'], 0)  
    for p in range(10, 110, 10) : 
        upper = np.percentile(x['begin_month'], p) 
        dataset.loc[((lower <= dataset['begin_month']) & (dataset['begin_month'] <= upper)), 'begin_month'] = p // 10
        lower = upper
    return dataset

def sampling_func(data, num_sample):
    N = len(data)
    sample = data.take(np.random.permutation(N)[:num_sample])
    return sample

def make_dataset(type) : 
    scaler = StandardScaler()
    trainset = pd.read_csv('train.csv')
    testset = pd.read_csv('test.csv')
    dataset = pd.concat([trainset, testset])
    index_retire = dataset.loc[dataset['occyp_type'].isnull() & (dataset['DAYS_EMPLOYED'] == 365243), 'occyp_type']
    index_notretire = dataset.loc[dataset['occyp_type'].isnull() & (dataset['DAYS_EMPLOYED'] != 365243), 'occyp_type']
    dataset.loc[index_retire.index, 'occyp_type'] = 'retire' 
    dataset.loc[index_notretire.index, 'occyp_type'] = 'notretire' 
    dataset.loc[(dataset.child_num>=4), 'child_num'] = 4
    # family_size is closely correlated with child_num so delete family_size column.
    dataset['child_num'].replace({0 : 'zero', 1:'one', 2 :'two', 3:'three', 4 : 'many'}, inplace=True)
    dataset = divide_age(dataset)
    dataset = divide_employed(dataset)
    dataset = divide_begin_month(dataset)
    del dataset['family_size']
    # delete unnessary variables
    del dataset['phone']
    del dataset['email']
    del dataset['FLAG_MOBIL']
    del dataset['index']
    scaler_whole = scaler.fit(pd.get_dummies(dataset.iloc[:,:-1]))

    if type == 'train' : 
        data_x_concat = dataset.loc[dataset['credit'].isnull()==False,:].iloc[:, :-1]
        data_y_concat = dataset.loc[dataset['credit'].isnull()==False,:].iloc[:, -1]
        data_x_dummy = pd.get_dummies(data_x_concat)
        data_x_scaled = scaler_whole.transform(data_x_dummy)
        return data_x_scaled, data_y_concat

    elif type == 'test' : 
        data_concat = dataset.loc[dataset['credit'].isnull()==True, :].iloc[:, :-1]
        data_x_dummy = pd.get_dummies(data_concat)
        data_x_scaled = scaler_whole.transform(data_x_dummy)
        return data_x_scaled