from joblib import load
from part import * 
import os
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dacon_credit prediction') 
    parser.add_argument("--num_ensemble", type=int, default=10)
    parser.add_argument("--model", type=str, default='xgboost')
    args = parser.parse_args()
    for num_iter in range(args.num_ensemble) : 
        print('[{}/{}]'.format(num_iter+1, args.num_ensemble))
        test_x = make_dataset('test')
        best_model = load('result_{}/model_{}'.format(args.model, num_iter)) 
        result = best_model.predict_proba(test_x)
        if num_iter == 0 : 
            concat_result = np.expand_dims(result, axis=0)
        else : 
            concat_result = np.concatenate([concat_result, np.expand_dims(result, axis = 0)], axis=0)
    
    submit = pd.DataFrame(np.mean(concat_result, axis = 0))
    if os.path.isfile('submit.csv') == True : 
        os.remove('submit.csv')
    submit.to_csv('submit.csv')