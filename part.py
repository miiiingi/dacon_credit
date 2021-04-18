import numpy as np
import sklearn
from sklearn import datasets 
from sklearn.preprocessing import StandardScaler 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.model_selection import GridSearchCV

def sampling_func(data, num_sample):
    N = len(data)
    sample = data.take(np.random.permutation(N)[:num_sample])
    return sample
def make_dataset(type) : 
    scaler = StandardScaler()
    trainset = pd.read_csv('train.csv')
    testset = pd.read_csv('test.csv')
    dataset = pd.concat([trainset, testset])
    index_nan = dataset.loc[dataset.occyp_type.isnull(), 'occyp_type']
    dataset.loc[index_nan.index, 'occyp_type'] = 'unknown' 
    dataset.loc[(dataset.child_num>=4), 'child_num'] = 4
    dataset['child_num'].replace({0 : 'zero', 1:'one', 2 :'two', 3:'three', 4 : 'many'}, inplace=True)
    # family_size is closely correlated with child_num so delete family_size column.
    del dataset['family_size']
    # delete unnessary variables
    del dataset['phone']
    del dataset['email']
    del dataset['FLAG_MOBIL']
    print(pd.get_dummies(dataset.iloc[:, :-1]).columns)
    scaler_whole = scaler.fit(pd.get_dummies(dataset.iloc[:,:-1]))
    if type == 'train' : 
        # prepare to datasets(down-sampling for considering imbalance data label)
        length_0 = dataset.loc[dataset['credit']==0, :].shape[0]
        data_0 = sampling_func(dataset.loc[dataset['credit']==0, :], length_0)
        data_1 = sampling_func(dataset.loc[dataset['credit']==1, :], length_0)
        data_2 = sampling_func(dataset.loc[dataset['credit']==2, :], length_0)
        data_concat = pd.concat([data_0, data_1, data_2])
        data_x_concat = data_concat.iloc[:, :-1]
        data_y_concat = data_concat.iloc[:, -1]
        data_x_dummy = pd.get_dummies(data_x_concat)
        print(data_x_dummy.columns)
        data_x_scaled = scaler_whole.transform(data_x_dummy)
        data_x_scaled = data_x_scaled[:, 1:] 
        return data_x_scaled, data_y_concat

    elif type == 'test' : 
        data_concat = dataset.loc[dataset['credit'].isnull(), :].iloc[:, :-1]
        data_x_dummy = pd.get_dummies(data_concat)
        data_x_scaled = scaler_whole.transform(data_x_dummy)
        data_x_scaled = data_x_scaled[:, 1:] 
        return data_x_scaled