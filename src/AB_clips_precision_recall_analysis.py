"""This script takes many ADAP format bounding box JSON files 
to evaluate the detections in precision and recall.
"""
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from ADAP_json_utils import precision_recall_eval
from ego4d_utils import collect_AB_clips

ignore_optional = False
plot_curve = False

### Estimate AB clips PR curves using ego4d GT
prefix = "APPEN"
base_dir = '/home/cyxu/hdd/ego4d_eval/clips_for_AB_testing_finally_selected/'
clips = collect_AB_clips(base_dir)
labels = [f'{prefix}_thresh{i}' for i in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]]
judgements_csv = ''

### shared configs
figure_name = f'{prefix}_precision_recall_thresholds_curves_ignore_Uncertain.png'
figure_title = f"Model-assisted Video Annotation Preicision-Recall ({prefix})"

mean_judgements_by_thres = []
steps = 1
thresholds = [0.7]

for thres in labels:
    gts, preds = [], []
    for clip_name, info in clips.items():
        test_json = os.path.join(base_dir, 'PR_analysis', thres, clip_name + '.json')
        
        pred = json.load(open(test_json))
        gt = json.load(open(info['ego4d']))

        gts.append(gt)
        preds.append(pred)

    # load and analyze all judgements
    precision, recall = precision_recall_eval(gts, preds, thresholds, ignore_optional)
    mean_judgements_by_thres.append([precision[0], recall[0]])

# plot precision-recall curve
fig, ax = plt.subplots(figsize=(6,6))

mean_judgements_by_thres = np.array(mean_judgements_by_thres)

for i in range(len(mean_judgements_by_thres)):
    x, y = mean_judgements_by_thres[i, 1], mean_judgements_by_thres[i, 0]
    if y < 1 or x < 1:
        continue
    ax.plot(x, y, 'o', label=f'{labels[i]},Pr:{round(y,1)}%,Rc:{round(x,1)}%')

# print stats to figure captions
# recall_mean, recall_std = round(np.mean(recalls), 2), round(np.std(recalls), 2)
# precision_mean, precision_std = round(np.mean(precisions), 2), round(np.std(precisions), 2)

# fig_caption = f"Recall mean: {recall_mean}, standard deviation: {recall_std} \n Precision mean: {precision_mean}, standard deviation: {precision_std}"
# # put left-aligned text above the legend
# fig.text(0.15, 0.55, fig_caption, fontsize=12, ha='left')

ax.set_xlabel('Recall (%)')
ax.set_ylabel('Precision (%)')
ax.legend(loc='lower left')
plt.xlim(75,85)
plt.ylim(85,95)
plt.grid()
plt.title(figure_title)
plt.savefig(base_dir+figure_name)