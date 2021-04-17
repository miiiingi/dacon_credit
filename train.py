import pandas as pd
import argparse
from scipy._lib.six import u
from sklearn.svm import SVC
from part import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC 
from joblib import dump, load
from hyperopt import tpe
from sklearn import * 
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval
import os
def fn_objective(params) : 
    classifier_type = params['type']
    del params['type']
    if classifier_type == 'svm' : 
        clf = SVC(random_state=722, probability = True, **params)
    elif classifier_type == 'xgboost' : 
        clf = XGBClassifier(random_state=722, **params)
    elif classifier_type == 'randomforest' : 
        clf = RandomForestClassifier(random_state=722, **params)
    else : 
        return 0
    loss = cross_val_score(clf, train_x, train_y, cv = 5, scoring = 'neg_log_loss').mean()
    return {'loss' : loss, 'status' : STATUS_OK} 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dacon_credit prediction') 
    parser.add_argument("--num_ensemble", type=int, default=10)
    args = parser.parse_args()
    if os.path.isdir('result_wholenormal') == False : 
        os.makedirs('result_wholenormal')
    for num_iter in range(args.num_ensemble) : 
        train_x, train_y = make_dataset('train')
        search_space = hp.choice('classifier_type', [
            {
                'type' : 'randomforest',
                'max_depth' : hp.choice('max_depth_rf', np.arange(1, 20, dtype=int)),
                'max_leaf_nodes' : hp.choice('max_leaf_nodes_rf', np.arange(1, 50, dtype=int)),
                'n_estimators' : hp.choice('n_estimators_rf', np.arange(10, 200, dtype = int))
            },
            {
                'type' : 'svm',
                'C' : hp.uniform('C_svm', 0.001, 10),
                'gamma' : hp.uniform('gamma_svm', 0.01, 10),
                'kernel' : hp.choice('kernel_svm', ['linear','rbf'])
            },
            {
                'type' : 'xgboost',
                'max_depth' : hp.choice('max_depth_xg', np.arange(3, 10, dtype=int)),
                'subsample' : hp.uniform('subsample_xg', 0.5, 1),
                'colsample_bytree' : hp.uniform('colsample_bytree', 0.5, 1),
                'n_estimators' : hp.choice('n_estimators', np.arange(10, 200, dtype = int)),
                'min_child_weight' : hp.choice('min_child_weight_xg', np.arange(3, 10, dtype = int)),
                'gamma' : hp.uniform('gamma_xg', 0, 0.5),
                'reg_alpha' : hp.uniform('reg_alpha', 0, 0.05),
            },
        ])

        best_result = fmin(fn = fn_objective, space = search_space, algo = tpe.suggest, max_evals=2)
        best_result = space_eval(search_space, best_result)
        type_model = best_result['type']
        del best_result['type']
        type_parameter = best_result
        if type_model == 'xgboost' : 
            clf = XGBClassifier(random_state=722, **type_parameter)
        elif type_model == 'svm' : 
            clf = SVC(random_state=722, probability = True, **type_parameter)
        elif type_model == 'randomforest' : 
            clf = RandomForestClassifier(random_state=722, **type_parameter)
        clf.fit(train_x, train_y)
        dump(clf, 'result_wholenormal/model_{}'.format(num_iter))