import numpy as np
from sklearn.preprocessing import StandardScaler 
import pandas as pd

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
    del dataset['index']
    scaler_whole = scaler.fit(pd.get_dummies(dataset.iloc[:,:-1]))
    if type == 'train' : 
        data_x_concat = dataset.loc[dataset['credit'].isnull()==False,:].iloc[:, :-1]
        data_y_concat = dataset.loc[dataset['credit'].isnull()==False,:].iloc[:, -1]
        data_x_dummy = pd.get_dummies(data_x_concat)
        data_x_scaled = scaler_whole.transform(data_x_dummy)
        # data_x_scaled = data_x_scaled[:, 1:] 
        return data_x_scaled, data_y_concat

    elif type == 'test' : 
        data_concat = dataset.loc[dataset['credit'].isnull()==True, :].iloc[:, :-1]
        data_x_dummy = pd.get_dummies(data_concat)
        data_x_scaled = scaler_whole.transform(data_x_dummy)
        # data_x_scaled = data_x_scaled[:, 1:] 
        return data_x_scaled