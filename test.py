from joblib import load
from part import * 
import argparse
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='dacon_credit prediction') 
    parser.add_argument("--num_ensemble", type=int, default=10)
    args = parser.parse_args()
    for num_iter in range(args.num_ensemble) : 
        test_x = make_dataset('test.csv', 'test')
        best_model = load('result/model_{}'.format(num_iter)) 
        result = best_model.predict_proba(test_x)
        if num_iter == 0 : 
            concat_result = np.expand_dims(result, axis=0)
        else : 
            concat_result = np.concatenate([concat_result, np.expand_dims(result, axis = 0)], axis=0)
    submit = pd.DataFrame(np.mean(concat_result, axis = 0))
    submit.to_csv('submit.csv')