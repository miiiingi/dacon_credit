from hpsklearn.components import sklearn_SVC
import numpy as np
import sklearn
from sklearn import datasets 
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_iris, fetch_openml 
from hpsklearn import HyperoptEstimator, extra_trees
import pandas as pd

def sampling_func(data, num_sample):
    N = len(data)
    sample = data.take(np.random.permutation(N)[:num_sample])
    return sample
def make_dataset(path, type) : 
    scaler = MinMaxScaler()
    if type == 'train' : 
        dataset = pd.read_csv(path)
        dataset_y = dataset.iloc[:, -1]
        dataset_x = dataset.iloc[:, :-1]   
        dataset_x.loc[(dataset_x.child_num>=4), 'child_num'] = 4
        dataset_x['child_num'].replace({0 : 'zero', 1:'one', 2 :'two', 3:'three', 4 : 'many'}, inplace=True)
        # family_size is closely correlated with child_num so delete family_size column.
        del dataset_x['family_size']
        # prepare to datasets
        length_0 = (dataset_y[dataset_y==0]).size
        data_x_0 = sampling_func(dataset_x.loc[dataset_y==0, :], length_0)
        data_x_1 = sampling_func(dataset_x.loc[dataset_y==1, :], length_0)
        data_x_2 = sampling_func(dataset_x.loc[dataset_y==2, :], length_0)
        data_y_0 = dataset_y[data_x_0.index]
        data_y_1 = dataset_y[data_x_1.index]
        data_y_2 = dataset_y[data_x_2.index]
        data_x_concat = pd.concat([data_x_0, data_x_1, data_x_2])
        data_y_concat = pd.concat([data_y_0, data_y_1, data_y_2])
        data_x_dummy = pd.get_dummies(data_x_concat)
        data_x_scaled = scaler.fit_transform(data_x_dummy)
        data_y_concat = data_y_concat.to_numpy()
        data_x_scaled = data_x_scaled[:, 1:] 
        return data_x_scaled, data_y_concat

    elif type == 'test' : 
        dataset = pd.read_csv(path)
        dataset.loc[(dataset.child_num>=4), 'child_num'] = 4
        dataset['child_num'].replace({0 : 'zero', 1:'one', 2 :'two', 3:'three', 4 : 'many'}, inplace=True)
        # family_size is closely correlated with child_num so delete family_size column.
        del dataset['family_size']
        # prepare to datasets
        data_x_dummy = pd.get_dummies(dataset)
        data_x_scaled = scaler.fit_transform(data_x_dummy)
        data_x_scaled = data_x_scaled[:, 1:] 
        return data_x_scaled

if __name__ == '__main__' :
    digits = fetch_openml('mnist_784', version = 1)
    X = digits.data
    y = digits.target

    test_size = int(0.2 * len(y))
    np.random.seed(13)
    indices = np.random.permutation(len(X))
    X_data = X[indices[:-test_size]]
    y_data = y[indices[:-test_size]]
    X_test = X[indices[-test_size:]]
    y_test = y[indices[-test_size:]]

    # Instantiate a HyperoptEstimator with the search space and number of evaluations

    estim = HyperoptEstimator(classifier=extra_trees('my_clf'),
                            preprocessing=[],
                            max_evals=10,
                            trial_timeout=300)

    # Search the hyperparameter space based on the data

    estim.fit( X_data, y_data )

    # Show the results

    print(estim.score(X_test, y_test))