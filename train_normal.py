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
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dacon_credit prediction') 
    parser.add_argument("--num_ensemble", type=int, default=10)
    parser.add_argument("--model", type=str, default='xgboost')

    args = parser.parse_args()
    if os.path.isdir('result_{}'.format(args.model)) == False : 
        os.makedirs('result_{}'.format(args.model))
    for num_iter in range(args.num_ensemble) : 
        train_x, train_y = make_dataset('train')
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