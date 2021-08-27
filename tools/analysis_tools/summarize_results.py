import os
import json
import pandas
from collections import defaultdict


def main():
    model_results_root_path = '/data/zhengwenhao/Result/DualCervixDetection/ModelResults/metric'

    metric_keys = ['AP_Top5', 'AP50_Top5', 'AP75_Top5', 'AP_Top10', 'AP50_Top10', 'AP75_Top10',
                   'AR_Top5', 'AR_Top10', 'FROC50', 'Recall50_fp_rate0.125', 'Recall50_fp_rate0.25',
                   'Recall50_fp_rate0.5', 'Recall50_fp_rate1', 'Recall50_fp_rate2', 'Recall50_fp_rate4',
                   'Recall50_fp_rate8', 'iRecall75_Top1', 'iRecall75_Top2', 'iRecall75_Top3']
    stage_names = ['acid', 'iodine']
    results = defaultdict(list)
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
                if 'dual' in exp_name:
                    for stage in stage_names:
                        cur_res = [exp_name, epoch_name]
                        for k in metric_keys:
                            if k in res['metric']:
                                cur_res.append(res['metric'][k])
                            elif k + '_' + stage in res['metric']:
                                cur_res.append(res['metric'][k + '_' + stage])
                            else:
                                raise KeyError
                        cur_res += [res_name[5:-5]]
                        results[stage].append(cur_res)
                elif any([stage in exp_name for stage in stage_names]):
                    for stage in stage_names:
                        if stage in exp_name:
                            cur_res = [exp_name, epoch_name]
                            for k in metric_keys:
                                if k + '_' + stage in res['metric']:
                                    cur_res.append(res['metric'][k + '_' + stage])
                                elif k in res['metric']:
                                    cur_res.append(res['metric'][k])
                                else:
                                    raise KeyError
                            cur_res += [res_name[5:-5]]
                            results[stage].append(cur_res)
                else:
                    raise RuntimeError(f'can not find the type of exp {exp_name}')
    with pandas.ExcelWriter(os.path.join(model_results_root_path, 'results.xlsx')) as writer:
        for stage in stage_names:
            df = pandas.DataFrame(results[stage],
                                  columns = ['exp name', 'epoch'] + [k + '_' + stage for k in metric_keys] + ['result name'])
            df.to_excel(writer, stage)


if __name__ == '__main__':
    main()
