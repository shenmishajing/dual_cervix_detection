import os
import json
import pandas


def main():
    model_results_root_path = '/path/to/results/'

    metric_keys = ['some', 'metric', 'keys']
    results = []
    for exp_name in os.listdir(model_results_root_path):
        exp_path = os.path.join(model_results_root_path, exp_name)
        if not os.path.isdir(exp_path):
            continue
        for epoch_name in os.listdir(exp_path):
            epoch_path = os.path.join(exp_path, epoch_name)
            if not os.path.isdir(epoch_path):
                continue
            for res_name in os.listdir(epoch_path):
                res_path = os.path.join(epoch_path, res_name)
                if not os.path.isfile(res_path) or not res_name.endswith('.json'):
                    continue
                res = json.load(open(os.path.join(res_path)))
                cur_res = [exp_name, epoch_name]
                for k in metric_keys:
                    if k in res['metric']:
                        cur_res.append(res['metric'][k])
                    else:
                        raise KeyError
                cur_res += [res_name[:-5]]
                results.append(cur_res)
    df = pandas.DataFrame(results, columns = ['exp name', 'epoch'] + metric_keys + ['result name'])
    df.to_csv(os.path.join(model_results_root_path, 'results.xlsx'))


if __name__ == '__main__':
    main()
