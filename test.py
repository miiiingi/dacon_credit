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
        best_model = load('result/model_{}'.format(num_iter)) 
        result = best_model.predict_proba(test_x)
        print(result, result.shape)
        exit()
        # dump(estimator, '{}_{}'.format(args.model, num_iter))