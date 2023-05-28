import os
import matplotlib.pyplot as plt
import json
import itertools
import math
results = {}

table_dict = {
    'exp04': 'Unc. SVDM',
    'exp05': 'SVDM/OE',
    'exp06': 'SVDM/SE',
    'exp08': 'SVDM/TE',
    'exp09': 'Unc. TVDM',
    'exp10': 'TVDM/OE',
    'exp11': 'TVDM/SE',
    'exp12': 'TVDM/TE',
    'exp14': 'SVDM/OE++',
    'exp15': 'TVDM/OE++',
    'exp16': 'MVDM/OE',
    'exp17': 'MVDM/SE',
    'exp18': 'MDVM/TE',
    'exp19': 'MVDM/OE++',
    'exp20': 'Unc. MVDM',
    'exp21': 'SVDM/FE',
    'exp22': 'TVDM/FE',
    'exp23': 'MVDM/FE',
}

metric_dict = {
    'fid': 'FID',
    'bpd': 'Bits per dimension',
    'diff_loss': 'Diffusion loss',
    'latent_loss': 'Latent loss',
    'encoder_loss': 'Encoder loss',
    'mi': 'Mutual information',
    'loss_recon': 'Reconstruction loss'
}



def get_best_metric(result, metric):
    best = result[0][metric]
    for r in result:
        if r[metric] > best:
            best = r[metric]
    return best

for path in os.listdir('results'):
    with open(os.path.join('results', path), 'r') as file:
        # Read each line of the file
        lines = file.readlines()

    # Initialize an empty list to store the dictionaries
    result = []
    # Iterate over each line and parse it as JSON
    for line in lines:
        # Parse the line as JSON and append the resulting dictionary to the list
        r = json.loads(line)
        result.append(r)

    results[os.path.splitext(path)[0]] = result

best_results = {}
for result in results:
    exp_name = table_dict[result]
    best_results[exp_name] = {}
    for metric in results[result][0].keys():
        if metric not in ['step', 'set', 'gamma_0', 'gamma_1']:
            #print(f"{result} {metric}: {get_best_metric(results[result], metric)}")
            best_results[exp_name][metric] = get_best_metric(results[result], metric)

d = best_results
best2 = dict([(x, dict([(k, d[k][x]) for k,v in d.items() if x in d[k]]))
            for x in set(itertools.chain(*[z for z in d.values()]))])

with open("best_results.json", "w") as outfile:
    json.dump(best_results, outfile)


# Make plots for each metric
for metric in best2.keys():
    if metric == 'mi':
        data = dict(sorted(best2[metric].items(), key=lambda x: x[1], reverse=True))
    else:
        data = dict(sorted(best2[metric].items(), key=lambda x: x[1], reverse=False))
    plt.figure(figsize=(7, 5))
    plt.bar(data.keys(), data.values())
    plt.title(metric_dict[metric])
    plt.xticks(rotation=25, ha="right")
    plt.subplots_adjust(bottom=0.15)
    plt.savefig(f"plots/{metric}.png")
