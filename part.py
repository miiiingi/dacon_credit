import numpy as np
import sklearn
from sklearn import datasets 
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.model_selection import GridSearchCV

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
        index_nan = dataset_x.loc[dataset_x.occyp_type.isnull(), 'occyp_type']
        dataset_x.loc[index_nan.index, 'occyp_type'] = 'unknown' 
        dataset_x.loc[(dataset_x.child_num>=4), 'child_num'] = 4
        dataset_x['child_num'].replace({0 : 'zero', 1:'one', 2 :'two', 3:'three', 4 : 'many'}, inplace=True)
        # family_size is closely correlated with child_num so delete family_size column.
        del dataset_x['family_size']
        # delete unnessary variables
        del dataset_x['phone']
        del dataset_x['email']
        del dataset_x['FLAG_MOBIL']
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
        data_x_scaled = data_x_scaled[:, 1:] 
        return data_x_scaled, data_y_concat

    elif type == 'test' : 
        dataset = pd.read_csv(path)
        index_nan = dataset.loc[dataset.occyp_type.isnull(), 'occyp_type']
        dataset.loc[index_nan.index, 'occyp_type'] = 'unknown' 
        dataset.loc[(dataset.child_num>=4), 'child_num'] = 4
        index_nan = dataset.loc[dataset.occyp_type.isnull(), 'occyp_type']
        dataset.loc[index_nan.index, 'occyp_type'] = 'unknown' 
        dataset['child_num'].replace({0 : 'zero', 1:'one', 2 :'two', 3:'three', 4 : 'many'}, inplace=True)
        # family_size is closely correlated with child_num so delete family_size column.
        del dataset['family_size']
        # delete unnessary variables
        del dataset['phone']
        del dataset['email']
        del dataset['FLAG_MOBIL']
        # prepare to datasets
        data_x_dummy = pd.get_dummies(dataset)
        data_x_scaled = scaler.fit_transform(data_x_dummy)
        data_x_scaled = data_x_scaled[:, 1:] 
        return data_x_scaled

def fn_objective(model, params, train_x, train_y) : 
    if model == 'randomforest' : 
        params= {'max_depth' : params['max_depth'], 'max_leaf_nodes' : params['max_leaf_nodes'], 'n_estimators' : params['n_estimators']}
        clf = RandomForestClassifier(random_state=722)
        # result_clf = GridSearchCV(clf, param_grid=params, cv = 5, scoring='neg_log_loss')
        result_clf = GridSearchCV(clf, param_grid = [], cv = 5, scoring='neg_log_loss')


        return result_clf.score(train_x, train_y) 
    # elif model == 'svm' : 
    #     gridsearch_svm = GridSearchCV(SVC(probability=True, random_state=722), param_grid=param_grid_svm, cv = 5, scoring='neg_log_loss')
    #     gridsearch_svm.fit(train_x, train_y)
    #     dump(gridsearch_svm, '{}/{}_{}'.format(args.model, args.model, num_iter))
    #     print(gridsearch_svm.score(train_x, train_y))
    # elif model == 'xgboost' : 
    #     gridsearch_xgboost = GridSearchCV(xgb.XGBClassifier(), param_grid=param_grid_svm, cv = 5, scoring='neg_log_loss')
    #     gridsearch_xgboost.fit(train_x, train_y)
    #     dump(gridsearch_xgboost, '{}/{}_{}'.format(args.model, args.model, num_iter))
    #     print(gridsearch_xgboost.score(train_x, train_y))
    # pass
