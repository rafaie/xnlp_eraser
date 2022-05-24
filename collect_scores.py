import argparse
from pathlib import Path
import os
import json
import csv 
import datetime

DEFAULT_EXPERIMENT='experiment_3'

# https://stackoverflow.com/questions/51359783/how-to-flatten-multilevel-nested-json
def flatten_data(y):
    out = {}

    def flatten(x, name=''):
        if type(x) is dict:
            for a in x:
                flatten(x[a], name + a + '_')
        elif type(x) is list:
            i = 0
            for a in x:
                flatten(a, name + str(i) + '_')
                i += 1
        else:
            out[name[:-1]] = x

    flatten(y)
    return out

def ext_metrics(exp:str):
    results = []

    # collect metrics
    base_path = f"{exp}/output"
    for p in Path(base_path).rglob('*_score.json'):
        p = str(p)
        d = json.load(open(p))
        d_new = flatten_data(d)
        d_new['1_experiment'] = exp
        d_new['2_dataset'] = p.split(os.sep)[2]
        d_new['3_config'] = p.split(os.sep)[3]
        d_new['4_date'] = p.split(os.sep)[4]
        d_new['5_metrics'] = os.path.basename(p)

        results.append(d_new)
    
    # find keys
    k = set()
    for l in results:
        k.update(l.keys())
    k = sorted(k)

    # save the files
    dt = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    with open(os.path.join('metrics', f'{exp}_{dt}.csv'), 'w') as fi: 
        writer = csv.writer(fi)
        writer.writerow(k)
        for l in results:
            r = []
            for k1 in k:
                r.append(l.get(k1))
            writer.writerow(r)

    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--experiment",
                        dest="experiment",
                        type=str,
                        default=DEFAULT_EXPERIMENT,
                        help="experiment Name. Example: experiment_3")
    args = parser.parse_args()
    ext_metrics(exp=args.experiment)