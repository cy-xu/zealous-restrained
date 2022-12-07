""" Check the face detector's performance on LaSOT dataset.
Find outliers/failed cases using the human bounding box.
"""
import os
from tracker.visualize import plot_tracking, plot_tracking_cues
from ego4d_utils import *
from ADAP_json_utils import ADAP_json_to_Tracks, ADAP_bbox_to_tlbr
from AB_expert_review import collect_all_judgements
import seaborn as sns
from tracker.yolox_matching import ious, linear_assignment


def json_xywh_to_tlbr_boxes(json_file):

    with open(json_file, 'r') as f:
        json_read = json.load(f)
        frames = json_read['annotation']['frames']

    tlbr_list = []
    boxes_total = 0

    for i in range(len(frames)):
        frm = str(i + 1)
        tlbr_curr = []

        for key, value in frames[frm]['shapesInstances'].items():
            # convert to top-left bottom-right
            x, y, w, h = value['x'], value['y'], value['width'], value['height']
            tlbr_curr.extend([[x, y, x+w, y+h]])
            boxes_total += 1

        tlbr_list.append(tlbr_curr)

    return tlbr_list, boxes_total


def pointplot(df, base_dir):

    ylabel = 'Actions/Decisions'
    order = ["Overall", "Easy", "Medium", "Hard"]
    group_order = ['A', 'B', 'C']
    decision_order = ['boxes_accepted', 'boxes_rejected', 'boxes_solved']
    
    # add a new Overall to Difficulty that includes all judgements
    df_copy = df.copy()
    df_copy['Difficulty'] = 'Overall'
    df_overall = pd.concat([df, df_copy])

    df_temp = df_overall[df_overall['Difficulty'] == 'Overall']

    df_changes_count = pd.melt(df, id_vars=['Group', 'Difficulty', 'tenure_group'], value_vars=decision_order, var_name='Decision', value_name='Count')
    
    # plt.figure()
    sns.set(rc={'figure.figsize':(9, 5)})
    fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True, gridspec_kw={'width_ratios': [3, 3]})

    # ax = sns.pointplot(data=df, hue='Group', y=label, dodge=True, capsize=.1, width=.5, ci=95, seed=0, join=True)

    # left figure, Overvall 
    # sns.barplot(ax=ax1, data=df_changes_count, x='Decision', order=decision_order, y='Count', hue='Group', hue_order=group_order, dodge=True, capsize=.1, ci=95, seed=0, join=False)
    sns.barplot(ax=ax1, data=df_changes_count, x='Decision', y='Count', hue='Group', capsize=.1, ci=95, seed=0)
    ax1.set(xlabel=None, ylabel=ylabel)
    ax1.get_legend().remove()

    # right figure, per Difficulty
    # sns.pointplot(ax=ax2, data=df, x='Difficulty', order=order[1:], hue_order=hue_order, y=label, hue='Group', dodge=True, capsize=.1, width=.5, ci=95, seed=0, join=True)
    sns.barplot(ax=ax2, data=df_changes_count, x='Difficulty', order=order[1:], y='Count', hue='Group', hue_order=group_order, capsize=.1, ci=95, seed=0)
    ax2.set(xlabel='Video Difficulty', ylabel=None)
    # ax2.get_legend().remove()
    # plt.legend(loc='lower right')

    # plt.ylim(0.0, 1.45)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f'fig_box_changes.pdf'))
    plt.savefig(os.path.join(base_dir, f'fig_box_changes.png'))

    # collect data for print
    # for group in ['A', 'B', 'C']:
    #     for difficulty in order:
    #         mean = np.mean(df[(df['Difficulty'] == difficulty) & (df['Group'] == group)][label])
    #         print(f'{part_label} {group} {difficulty} mean: {mean:.2f}')

    plt.clf() # this clears the figure
    print(f'Plotting done.')

if __name__ == "__main__":

    review_part2 = False
    NO_CHANGE_IOU = 0.95
    Consensus_thresh = 0.75
    fps = 30 # frames per second

    base_dir = 'user_study_results/'

    if review_part2:
        part_dir = os.path.join(base_dir, 'part_2_merged_judgements')
    else:
        part_dir = os.path.join(base_dir, 'part_1_merged_judgements')
    
    df_path = os.path.join(part_dir, 'AB_testing_precision_recall_IOU_0.5.csv')
    df_new_path = os.path.join(part_dir, 'AB_testing_precision_recall_IOU_0.5_with_changes_count.csv')
    df = pd.read_csv(df_path)

    #  if changes count already processed, plot the figures
    if os.path.isfile(df_new_path):
        df = pd.read_csv(df_new_path)

        pointplot(df, part_dir)

        breakpoint()

    # main loop to process each video clip
    for i, row in df.iterrows():

        print(f'processing judgement {i}')
        boxes_accepted = 0
        boxes_rejected = 0
        boxes_solved = 0

        # final judgements created/reviewed by the raters
        rater_json = row['Rater_json']
        gt_json = row['GT_json']

        rater_boxes, rater_total = json_xywh_to_tlbr_boxes(rater_json)

        if row['Group'] == 'A':
            boxes_accepted = 0
            boxes_rejected = 0
            boxes_solved = rater_total
            init_total = 0
        else:
            if row['Group'] == 'B':
                init_json = row['Byte_json']
            elif row['Group'] == 'C':
                init_json = row['Appen_json']

            init_boxes, init_total = json_xywh_to_tlbr_boxes(init_json)

            # check each frame for changes
            for j in range(len(rater_boxes)):
                rater_curr = rater_boxes[j]
                init_curr = init_boxes[j]

                # cython_bbox
                _ious = ious(init_curr, rater_curr)
                iou_cost_matrix = 1 - _ious

                matches, u_init, u_rater = linear_assignment(iou_cost_matrix, thresh=NO_CHANGE_IOU)

                boxes_accepted += len(matches)
                boxes_solved += len(u_rater)
                boxes_rejected += len(u_init)

        # write the stats to the csv
        df.loc[i, 'init_total'] = init_total
        df.loc[i, 'rater_total'] = rater_total
        df.loc[i, 'boxes_accepted'] = boxes_accepted
        df.loc[i, 'boxes_rejected'] = boxes_rejected
        df.loc[i, 'boxes_solved'] = boxes_solved

    # save the changes to the csv
    df.to_csv(df_new_path, index=False)

        # stats
        # ratio = optional_total/box_total
        # print(f'{optional_total} boxes marked optional, {ratio*100:.2f}%')