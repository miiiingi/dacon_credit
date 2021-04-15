from joblib.logger import PrintTime
import pandas as pd
import argparse
from scipy._lib.six import u
from sklearn.svm import SVC
from part import *
from hpsklearn import HyperoptEstimator, extra_trees, svc, random_forest, any_classifier
from sklearn.metrics import log_loss, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC 
from joblib import dump, load
from hyperopt import tpe
from sklearn import * 
import xgboost as xgb
from hyperopt import fmin, tpe, hp
import os
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dacon_credit prediction') 
    parser.add_argument("--num_ensemble", type=int, default=10)
    parser.add_argument("--model", type=str, default='randomforest')
    args = parser.parse_args()
    if os.path.isdir('{}'.format(args.model)) == False : 
        os.makedirs('{}'.format(args.model))
    param_grid_svm= {'C' : [0.001, 0.1, 100],'gamma' : [0.001, 1]}
    param_grid_randomforest= {'max_depth' : hp.uniform('max_depth',0, 15), 'max_leaf_nodes' : hp.uniform('max_leaf_nodes', 0, 50),'n_estimators' : hp.uniform('n_estimators', 0, 200)}
    param_grid_xgboost= {'max_depth' : [5, 10, 15], 'max_leaf_nodes' : [20, 40, 60, 80], 'n_estimators' : [50, 100, 150, 200]}

    for num_iter in range(args.num_ensemble) : 
        train_x, train_y = make_dataset('train.csv', 'train')
        if args.model == 'randomforest' : 
            model_best = fmin(fn = fn_objective(args.model, param_grid_randomforest, train_x, train_y), space=param_grid_randomforest, max_evals=10, rstate=np.random.RandomState(722), algo = tpe.suggest)
            dump(model_best, '{}/{}_{}'.format(args.model, args.model, num_iter))
            print(model_best.score(train_x, train_y))
            exit()
        elif args.model == 'svm' : 
            gridsearch_svm = GridSearchCV(SVC(probability=True, random_state=722), param_grid=param_grid_svm, cv = 5, scoring='neg_log_loss')
            gridsearch_svm.fit(train_x, train_y)
            dump(gridsearch_svm, '{}/{}_{}'.format(args.model, args.model, num_iter))
            print(gridsearch_svm.score(train_x, train_y))
        elif args.model == 'xgboost' : 
            gridsearch_xgboost = GridSearchCV(xgb.XGBClassifier(), param_grid=param_grid_svm, cv = 5, scoring='neg_log_loss')
            gridsearch_xgboost.fit(train_x, train_y)
            dump(gridsearch_xgboost, '{}/{}_{}'.format(args.model, args.model, num_iter))
            print(gridsearch_xgboost.score(train_x, train_y))