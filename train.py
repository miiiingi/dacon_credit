import pandas as pd
import argparse
from scipy._lib.six import u
from sklearn.svm import SVC
from part import *
from hpsklearn import HyperoptEstimator, extra_trees, svc, random_forest, any_classifier
from sklearn.metrics import log_loss, make_scorer
from joblib import dump, load
from hyperopt import tpe
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dacon_credit prediction') 
    parser.add_argument("--num_ensemble", type=int, default=10)
    parser.add_argument("--model", type=str, default='randomforest')
    args = parser.parse_args()
    for num_iter in range(args.num_ensemble) : 
        train_x, train_y = make_dataset('train.csv', 'train')
        # 일단 제출해보고 결과가 나쁘면 loss fn 건드려 보자
        estimator = HyperoptEstimator(classifier=random_forest('clf'), preprocessing=[], algo=tpe.suggest, max_evals= 30,  continuous_loss_fn=False)
        estimator.fit(train_x, train_y)
        dump(estimator, '{}_{}'.format(args.model, num_iter))