import numpy as np
from numpy.lib.shape_base import column_stack
from pandas.core.tools.datetimes import DatetimeScalar
from scipy.sparse import data
from sklearn.preprocessing import StandardScaler, OneHotEncoder
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

def scaling(dataset, columns) :  
    for column in columns : 
        dataset[column] = (dataset[column] - min(dataset[column])) / (max(dataset[column]) - min(dataset[column]))
    return dataset

# def make_dataset(type) : 
#     scaler = StandardScaler()
#     trainset = pd.read_csv('train.csv')
#     testset = pd.read_csv('test.csv')
#     dataset = pd.concat([trainset, testset])

#     # columns_numerical = ['income_total', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'begin_month']
#     # index_retire = dataset.loc[dataset['occyp_type'].isnull() & (dataset['DAYS_EMPLOYED'] == 365243), 'occyp_type']
#     # index_notretire = dataset.loc[dataset['occyp_type'].isnull() & (dataset['DAYS_EMPLOYED'] != 365243), 'occyp_type']
#     dataset.loc[dataset.loc[dataset['occyp_type'].isnull(), 'occyp_type'].index, 'occyp_type'] = 'unknown' 
#     # dataset.loc[index_retire.index, 'occyp_type'] = 'retire' 
#     # dataset.loc[index_notretire.index, 'occyp_type'] = 'notretire' 
#     dataset.loc[dataset['child_num'] >= 5, 'child_num'] = 5
#     # dataset['DAYS_BIRTH'] = (-(dataset['DAYS_BIRTH']) // 30)
#     # dataset['DAYS_EMPLOYED'] = (-(dataset['DAYS_EMPLOYED']) // 30)
#     # dataset.loc[(dataset['DAYS_EMPLOYED'] == 0), 'DAYS_EMPLOYED'] = max(dataset['DAYS_EMPLOYED']) 
#     # dataset['begin_month'] = (-(dataset['begin_month']))
#     # dataset = scaling(dataset, columns_numerical)
#     # dataset.loc[(dataset['DAYS_EMPLOYED'] == 0), 'DAYS_EMPLOYED'] = max(dataset['DAYS_EMPLOYED'])
#     # family_size is closely correlated with child_num so delete family_size column.
#     # dataset['child_num'].replace({0 : 'zero', 1:'one', 2 :'two', 3:'three', 4 : 'many'}, inplace=True)
#     # dataset = divide_age(dataset)
#     # dataset = divide_employed(dataset)
#     # dataset = divide_begin_month(dataset)
#     # delete unnessary variables
#     # scaler_whole = scaler.fit(pd.get_dummies(dataset.iloc[:,:-1]).loc[:,['income_total','DAYS_BIRTH','DAYS_EMPLOYED','begin_month']])
#     scaler_whole = scaler.fit(pd.get_dummies(dataset.loc[:,dataset.columns != 'credit']))

#     if type == 'train' : 
#         data_x_concat = dataset.loc[dataset['credit'].isnull()==False,:].loc[:,dataset.columns != 'credit']
#         data_y_concat = dataset.loc[dataset['credit'].isnull()==False,:].loc[:, 'credit']
#         data_x_dummy = pd.get_dummies(data_x_concat)
#         data_x_scaled = scaler_whole.transform(data_x_dummy)
#         # data_x_scaled = scaler_whole.transform(data_x_dummy.iloc[:,:-1].loc[:,['income_total','DAYS_BIRTH','DAYS_EMPLOYED','begin_month']])
#         return data_x_scaled, data_y_concat

#     elif type == 'test' : 
#         data_concat = dataset.loc[dataset['credit'].isnull()==True, :].loc[:,dataset.columns != 'credit']
#         data_x_dummy = pd.get_dummies(data_concat)
#         data_x_scaled = scaler_whole.transform(data_x_dummy)
#         # data_x_scaled = scaler_whole.transform(data_x_dummy.iloc[:,:-1].loc[:,['income_total','DAYS_BIRTH','DAYS_EMPLOYED','begin_month']])
#         return data_x_scaled
def make_dataset(type) : 
    enc = OneHotEncoder()
    trainset = pd.read_csv('train.csv')
    trainset.drop('index', axis = 1, inplace=True)
    trainset.loc[trainset.loc[trainset['occyp_type'].isnull(), 'occyp_type'].index, 'occyp_type'] = 'unknown' 
    trainset['identity'] = [str(i) + str(j) + str(k) + str(l) + str(m) for i,j,k,l,m in zip(trainset['gender'],trainset['income_total'],trainset['income_type'],trainset['DAYS_BIRTH'],trainset['DAYS_EMPLOYED'])] #
    for index, value in zip(trainset['identity'].value_counts().index, trainset['identity'].value_counts()) : #
        if value > 1 : 
            trainset.loc[trainset['identity'] == index, 'derivatives_begin_month'] = 0 
        else :  
            trainset.loc[trainset['identity'] == index, 'derivatives_begin_month'] = 1 
    trainset.drop('identity', axis = 1, inplace=True) #
    object_col = [] 
    for col in trainset.columns : 
        if trainset[col].dtype == 'object' : 
            object_col.append(col)
    enc.fit(trainset.loc[:, object_col])
    trainset_onehot = pd.DataFrame(enc.transform(trainset.loc[:,object_col]).toarray(), columns=enc.get_feature_names(object_col))
    trainset.drop(object_col, axis= 1, inplace=True)
    trainset_concat = pd.concat([trainset, trainset_onehot], axis= 1)

    # trainset_concat.loc[trainset_concat.duplicated(), 'duplicated'] = 1
    # trainset_concat.loc[trainset_concat.duplicated() == False, 'duplicated'] = 0 
    testset = pd.read_csv('test.csv')
    testset.drop('index', axis = 1, inplace=True)
    testset.loc[testset.loc[testset['occyp_type'].isnull(), 'occyp_type'].index, 'occyp_type'] = 'unknown' 
    testset['identity'] = [str(i) + str(j) + str(k) + str(l) + str(m) for i,j,k,l,m in zip(testset['gender'],testset['income_total'],testset['income_type'],testset['DAYS_BIRTH'],testset['DAYS_EMPLOYED'])] #
    for index, value in zip(testset['identity'].value_counts().index, testset['identity'].value_counts()) : #
        if value > 1 : 
            testset.loc[testset['identity'] == index, 'derivatives_begin_month'] = 0  
        else :  
            testset.loc[testset['identity'] == index, 'derivatives_begin_month'] = 1 
    testset.drop('identity', axis = 1, inplace=True) #

    object_col = [] 
    for col in testset.columns : 
        if testset[col].dtype == 'object' : 
            object_col.append(col)
    enc = OneHotEncoder()
    enc.fit(testset.loc[:, object_col])
    testset_onehot = pd.DataFrame(enc.transform(testset.loc[:,object_col]).toarray(), columns=enc.get_feature_names(object_col))
    testset.drop(object_col, axis= 1, inplace=True)
    testset_concat = pd.concat([testset, testset_onehot], axis= 1)
    # testset_concat.loc[testset_concat.duplicated(), 'duplicated'] = 1
    # testset_concat.loc[testset_concat.duplicated() == False, 'duplicated'] = 0 

    if type == 'train' : 
        data_x = trainset_concat.loc[:,trainset_concat.columns != 'credit']
        data_y = trainset_concat.loc[:, 'credit']
        return data_x, data_y

    elif type == 'test' : 
        return testset_concat 