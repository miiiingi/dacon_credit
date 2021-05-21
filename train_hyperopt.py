import pandas as pd
import argparse
from scipy._lib.six import u
from sklearn.svm import SVC
from part import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.svm import SVC 
from sklearn.feature_selection import RFE
from joblib import dump, load
from hyperopt import tpe
from sklearn import * 
from xgboost import XGBClassifier
from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval
from lightgbm import LGBMClassifier
import os
def fn_objective(params) : 
    classifier_type = params['type']
    del params['type']
    if classifier_type == 'lightgbm' : 
        clf = LGBMClassifier(random_state=722, **params)
    elif classifier_type == 'xgboost' : 
        clf = XGBClassifier(random_state=722, objective = 'multi:softprob', use_label_encoder=False, **params)
    else : 
        return 0
    fold = StratifiedKFold(n_splits = 5, shuffle =True, random_state = 722)
    loss = cross_val_score(clf, train_x, train_y, cv = fold, scoring = 'neg_log_loss').mean()

    return {'loss' : -loss, 'status' : STATUS_OK} 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dacon_credit prediction') 
    parser.add_argument("--num_ensemble", type=int, default=10)
    parser.add_argument("--model", type=str, default='xgboost')

    args = parser.parse_args()
    if os.path.isdir('result_{}'.format(args.model)) == False : 
        os.makedirs('result_{}'.format(args.model))
    for num_iter in range(args.num_ensemble) : 
        train_x, train_y = make_dataset('train')
        search_space = hp.choice('classifier_type', [
            # {
            #     'type' : 'lightgbm',
            #     'max_depth' : hp.choice('max_depth_lightgbm', np.arange(1, 100, dtype=int)),
            #     'min_child_samples' : hp.choice('min_child_samples_lightgbm', np.arange(1, 100, dtype=int)),
            #     'subsample' : hp.uniform('subsample_lightgbm', 0.5, 1),
            #     'n_estimators' : hp.choice('n_estimators_lightgbm', np.arange(100, 1000, dtype = int))
            # },
            {
                'type' : 'xgboost',
                # 'predictor' : 'gpu_predictor',
                # 'tree_method' : 'gpu_hist',
                'eval_metric' : 'mlogloss', 
                'max_depth' : hp.choice('max_depth_xg', np.arange(3, 10, dtype=int)),
                'subsample' : hp.uniform('subsample_xg', 0.5, 1),
                'colsample_bytree' : hp.uniform('colsample_bytree_xg', 0.5, 1),
                'n_estimators' : hp.choice('n_estimators_xg', np.arange(10, 200, dtype = int)),
                'min_child_weight' : hp.choice('min_child_weight_xg', np.arange(3, 10, dtype = int)),
                'gamma' : hp.uniform('gamma_xg', 0, 0.5),
                'reg_alpha' : hp.uniform('reg_alpha', 0, 0.05),
                'eta' : hp.uniform('eta_xg', 0.01, 0.4)
            },
        ])

        best_result = fmin(fn = fn_objective, space = search_space, algo = tpe.suggest, max_evals = 128)
        best_result = space_eval(search_space, best_result)
        type_model = best_result['type']
        del best_result['type']
        if type_model == 'xgboost' : 
            clf = XGBClassifier(random_state=722, use_label_encoder=False, **best_result)
        elif type_model == 'lightgbm' : 
            clf = LGBMClassifier(random_state=722, **best_result)
        elif type_model == 'randomforest' : 
            clf = RandomForestClassifier(random_state=722, **best_result)
        clf.fit(train_x, train_y)
        dump(clf, 'result_{}/model_{}'.format(args.model, num_iter))
        del train_x
        del train_y