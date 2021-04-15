from hpsklearn import HyperoptEstimator, extra_trees, svc, random_forest
from joblib import dump, load
from part import * 
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dacon_credit prediction') 
    parser.add_argument("--num_ensemble", type=int, default=10)
    parser.add_argument("--model", type=str, default='randomforest')
    args = parser.parse_args()
    for num_iter in range(args.num_ensemble) : 
        test_x = make_dataset('test.csv', 'test')
        print(test_x[:5, :])
        exit()
        # 일단 제출해보고 결과가 나쁘면 loss fn 건드려 보자
        estimator = load('{}_{}'.format(args.model, num_iter)) 
        result = estimator.predict_proba(test_x)
        print(result, result.shape)
        exit()
        # dump(estimator, '{}_{}'.format(args.model, num_iter))