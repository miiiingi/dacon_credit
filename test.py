from joblib import load
from part import * 
import os
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dacon_credit prediction') 
    parser.add_argument("--num_ensemble", type=int, default=10)
    parser.add_argument("--model", type=str, default='xgboost')
    args = parser.parse_args()
    submit = pd.read_csv('sample_submission.csv', index_col = ['index'])
    for num_iter in range(args.num_ensemble) : 
        print('[{}/{}]'.format(num_iter+1, args.num_ensemble))
        test_x = make_dataset('test')
        best_model = load('result_{}/model_{}'.format(args.model, num_iter)) 
        submit.iloc[:, :] += best_model.predict_proba(test_x) / args.num_ensemble 
    submit.to_csv('submit.csv')