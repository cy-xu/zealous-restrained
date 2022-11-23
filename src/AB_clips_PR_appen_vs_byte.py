"""This script takes many ADAP format bounding box JSON files 
to evaluate the detections in precision and recall.
"""
import os
import shutil
import json
import numpy as np
import matplotlib.pyplot as plt
from ADAP_json_utils import precision_recall_eval
from ego4d_utils import collect_AB_clips

ignore_optional = False
show_diff = True

# thresholds used in linear_assignment, thus
# IOU_thres = 1-thresholds
steps = 1
thresholds = [0.7]

### Appen optimize higher thershold
prefix = f"Appen_vs_Byte_precision_recall_include_Uncertain_{thresholds[0]}_diff"
base_dir = '/home/cyxu/hdd/ego4d_eval/clips_for_AB_testing_finally_selected/'
byte_dir = '/home/cyxu/hdd/ego4d_eval/clips_for_AB_testing_processed_BYTE/'
appen_dir = '/home/cyxu/hdd/ego4d_eval/clips_for_AB_testing_processed_Appen/'

figure_name = f'{prefix}.png'
figure_title = f"{prefix} (36 * 30s clips)"
clips = collect_AB_clips(base_dir)

### shared configs


# make a copy of BYTE json to the shared directory, one time only

for root, dirs, files in os.walk(appen_dir):
    for file in files:
        if file.endswith('.mp4'):
            clip_name = file.split('.')[0]
            if clip_name in clips:
                print(clip_name)
                json_name = clip_name.split('_')[0]
                # shutil.copy(os.path.join(root, json_name+'.json'), os.path.join(base_dir, clips[clip_name]['key'], clip_name+'_BYTE.json'))
                # shutil.copy(os.path.join(root, json_name+'_gt.json'), os.path.join(base_dir, clips[clip_name]['key'], clip_name+'_EGO4D.json'))
                shutil.copy(os.path.join(root, clip_name+'_bbox.mp4'), os.path.join(base_dir, clips[clip_name]['key'], clip_name+'_APPEN.mp4'))
                # shutil.copy(os.path.join(root, clip_name+'_bbox.mp4'), os.path.join(base_dir, clips[clip_name]['key'], clip_name+'_BYTE.mp4'))

breakpoint()

# load and analyze all judgements from participants
judgements_appen = []
judgements_byte = []

for clip in clips:
    with open(clips[clip]['appen'], 'r') as f:
        appen = json.load(f)

    with open(clips[clip]['byte'], 'r') as f:
        byte = json.load(f)

    with open(clips[clip]['ego4d'], 'r') as f:
        ego4d = json.load(f)

    pre_appen, rec_appen = precision_recall_eval(ego4d, appen, steps, thresholds, ignore_optional)
    pre_byte, rec_byte = precision_recall_eval(ego4d, byte, steps, thresholds, ignore_optional)

    judgements_appen.append([pre_appen[0], rec_appen[0]])
    judgements_byte.append([pre_byte[0], rec_byte[0]])

# plot precision-recall curve
fig, ax = plt.subplots(figsize=(10,6))

judgements_appen = np.array(judgements_appen)
judgements_byte = np.array(judgements_byte)

# plot two curves
for i in range(len(judgements_appen)):
    recall = judgements_appen[i,1]
    precision = judgements_appen[i,0]
    ax.annotate(str(i), (recall, precision))

    if show_diff:
        recall = judgements_appen[i,1] - judgements_byte[i,1]
        precision = judgements_appen[i,0] - judgements_byte[i,0]
        ax.annotate(str(i), (recall, precision))

        if i == 0:
            ax.scatter(recall, precision, marker='o', label='APPEN-BYTE')
        else:
            ax.scatter(recall, precision, marker='o')
        continue

    if i == 0:
        ax.scatter(recall, precision, color='red', marker='o', label='APPEN')
    else:
        ax.scatter(recall, precision, color='red', marker='o')

    recall = judgements_byte[i,1]
    precision = judgements_byte[i,0]
    ax.annotate(str(i), (recall, precision))

    if i == 0:
        ax.scatter(recall, precision, color='blue', marker='o', label='BYTE')
    else:
        ax.scatter(recall, precision, color='blue', marker='o')
    
# precision, recall = judgements_appen[:, 0], judgements_appen[:, 1]
# ax.plot(precision, recall, label=f'APPEN, mean precision {np.mean(precision)}, mean recall {np.mean(recall)}', color='red')

# precision, recall = judgements_byte[:, 0], judgements_byte[:, 1]
# ax.plot(precision, recall, label=f'BYTE, mean precision {np.mean(precision)}, mean recall {np.mean(recall)}', color='blue')

# print stats to figure captions
appen_precision, appen_recall = round(np.mean(judgements_appen[:, 0]), 2), round(np.mean(judgements_appen[:, 1]), 2)
byte_precision, byte_recall = round(np.mean(judgements_byte[:, 0]), 2), round(np.mean(judgements_byte[:, 1]), 2)

fig_caption = f"APPEN Precision: {appen_precision}, Recall: {appen_recall} \n BYTE Precision: {byte_precision}, Recall: {byte_recall}"
# put left-aligned text above the legend
fig.text(0.15, 0.55, fig_caption, fontsize=12, ha='left')

ax.set_xlabel('Recall (%)')
ax.set_ylabel('Precision (%)')
ax.legend(loc='lower right')
# plt.xlim(0,100)
# plt.ylim(0,100)
if not show_diff:
    plt.yscale('log', base=10)
    plt.xscale('log', base=10)
plt.grid()
plt.title(figure_title)
plt.savefig(base_dir+figure_name)