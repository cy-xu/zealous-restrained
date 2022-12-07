"""This script ADAP results CSV files to evaluate the time for each video clip or group.
"""
import os
from turtle import width
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from ego4d_utils import *
import urllib.request

import seaborn as sns
# ANOVA table using statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols
import pingouin as pg

BASE_DIR = 'user_study_results/'
OUTLIER_FACTOR = 3

# Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

old_difficulty_dict = {
        '6c1bac5c-aead-40b8-9c17-be97d86b68d4_segment1': 'Easy',
        '5df66408-7a8c-43bb-9235-d088aa3d19dc_segment7': 'Easy',
        '50d3035c-2503-4bcf-b28b-e66c92c98f09_segment2': 'Easy',
        '7b76be35-4e6d-42d7-9fa9-14a21fc327cf_segment6': 'Easy',
        '5c8f8839-4bf9-45e0-beb0-6c17a5ead0d5_segment0': 'Easy',
        '5eaa3bfb-cc71-43fa-88dc-d367ef929a7b_segment4': 'Easy',
    '6c1bac5c-aead-40b8-9c17-be97d86b68d4_segment9': 'Easy',
    'dfe025f3-573e-4297-b17a-30e990cfa310_segment6': 'Easy',
    '19b880da-910e-40c7-970e-cd8718f2faf5_segment6': 'Easy',
    '36e7b2ad-dddd-42fa-a427-6411ff12f9b9_segment1': 'Easy',

        '2e651150-9338-4103-b231-2f089aa19df7_segment9': 'Easy',
        '70e88e82-5c06-44fd-959c-d4b2d40999cd_segment8': 'Easy',
        '70e88e82-5c06-44fd-959c-d4b2d40999cd_segment1': 'Easy',
        'be8f4445-3958-4fc4-8cdf-b14d1d0eba73_segment1':'Medium',
        'b0c57748-65c8-49d7-9681-ddcfcde3298b_segment4':'Medium',
        '8deeeb88-19ed-4747-9c79-d94104bed7e0_segment1':'Medium',
        '5b3d66ec-9569-4aca-a548-1521bcdc6b25_segment2':'Medium',
        '1261ee8c-80fc-4a3f-9dd3-29f4a918a9fe_segment7':'Medium',
        'b0c57748-65c8-49d7-9681-ddcfcde3298b_segment0':'Medium',
    '1261ee8c-80fc-4a3f-9dd3-29f4a918a9fe_segment5': 'Medium',
    'f55f2dfd-704f-42cf-a39e-72e2560314b3_segment6': 'Medium',
    'd5374572-5576-4f83-99c8-a0b12c4de160_segment2': 'Medium',
    '2e651150-9338-4103-b231-2f089aa19df7_segment0': 'Medium',
    'd0af657d-9c21-447c-a4f9-199b38deac5b_segment1': 'Medium',
    '33fd4f1f-fb01-4d3a-9b8d-b8d8024a2dab_segment5': 'Medium',

        '3ce15244-5293-4a5e-8c92-52ab29c6949a_segment2': 'Hard',
        '55363f67-1b41-476f-a83e-7e9741786ad7_segment2': 'Hard',
        'c4392ca6-4706-4a96-a6b2-29e6ef544500_segment6': 'Hard',
        '136cef9c-e17f-49f9-ae45-a17e7e5c5477_segment2': 'Hard',
        'd0af657d-9c21-447c-a4f9-199b38deac5b_segment8': 'Hard',
        '1261ee8c-80fc-4a3f-9dd3-29f4a918a9fe_segment0': 'Hard',
        '6640a311-86c4-4de7-964c-e3a4a67a3e02_segment5': 'Hard',
        '53144267-6da5-40d3-91f2-eaa85be367de_segment5': 'Hard',
        'd5374572-5576-4f83-99c8-a0b12c4de160_segment4': 'Hard',

    'be8f4445-3958-4fc4-8cdf-b14d1d0eba73_segment7': 'Hard',
    'd0af657d-9c21-447c-a4f9-199b38deac5b_segment0': 'Hard',
    'd0af657d-9c21-447c-a4f9-199b38deac5b_segment6': 'Hard',
    '1f12c871-9b3a-4611-a2b4-9c39059052a4_segment5': 'Hard',
    }

def get_judgement_time(row):
    # get start/stop times for each judgement
    delta = timedelta(hours=8)
    total_frames = 900
    detection_900_frms_overhead = 77 # seconds

    start_t = row['_started_at']
    start_t = datetime.strptime(start_t, '%m/%d/%Y %H:%M:%S')
    stop_t = row['_created_at']
    stop_t = datetime.strptime(stop_t, '%m/%d/%Y %H:%M:%S')

    # in seconds
    duration = (stop_t - start_t).total_seconds()
    if row['Group'] in ['B', 'C']: duration += detection_900_frms_overhead
    # in hours
    duration = duration/60./60.

    # speed as in frames/hour
    speed = total_frames / duration

    # efficiency as in seconds/frame
    efficiency = (stop_t - start_t).total_seconds() / total_frames

    # convert time zone from UTC to GMT+8
    start_t = start_t.replace(tzinfo=timezone.utc) + delta
    stop_t = stop_t.replace(tzinfo=timezone.utc) + delta

    # print(f'extra long job {unit_id}, local time: {start_t.strftime("%Y-%m-%d %H:%M")} to {stop_t.strftime("%Y-%m-%d %H:%M")}, duration: {int(duration/60)} hours')

    return start_t, round(duration, 3), round(speed, 2), round(efficiency, 2)

def reject_outliers(data, m=2):
    # if not np array, convert to np array
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    return data[abs(data - np.mean(data)) < m * np.std(data)]

def add_time_and_speed(df_new):
    # add speed and time for each row
    for idx, row in df_new.iterrows():
        start_time, duration, speed, efficiency = get_judgement_time(row)
        df_new.at[idx, 'Start_time'] = start_time
        df_new.at[idx, 'Duration'] = duration
        df_new.at[idx, 'Speed'] = speed
        df_new.at[idx, 'Efficiency'] = efficiency
    return df_new

def add_difficulty(df_new, part2=False):
    # add difficulty level label
    # find the row that include clip in the video_url column
    levels = ['Easy', 'Medium', 'Hard']
    difficulty_dict = {}

    # if old_difficulty:
    #     for idx, row in df_new.iterrows():
    #         df_new.at[idx, 'Difficulty'] = old_difficulty_dict[row['Clip_name']]
    #         # df_new.at[idx, 'Faces_per_frame'] = difficulty_dict[row['Clip_name']][1]
    #     return df_new

    if part2:
        gt_dir = BASE_DIR + 'part_2_merged_judgements/reviewed_judgements_v3/'
    else:
        gt_dir = BASE_DIR + 'part_1_merged_judgements/reviewed_judgements_v2/'
    
    clips = df_new['Clip_name'].unique()

    easy, medium, hard = 0, 0, 0
    for clip in clips:
        gt_json = gt_dir + f'{clip}_reviewed.json'

        with open(gt_json) as f: gt_json = json.load(f)
        non_empty_total, faces_total = 0, 0

        for frm in range(1, 901):
            
            # one video in Part 2 missed a single frame
            if str(frm) not in gt_json['annotation']['frames']:
                print(f'{clip} does not have frame {frm}')
                continue

            boxes = gt_json['annotation']['frames'][str(frm)]['shapesInstances']
            if len(boxes) > 0:
                non_empty_total += 1
                faces_total += len(boxes)

        effective_faces_per_frame = faces_total / non_empty_total
        if effective_faces_per_frame < 1.5:
            difficulty_dict[clip] = ['Easy', effective_faces_per_frame]
            easy += 1
        elif effective_faces_per_frame < 2.5:
            difficulty_dict[clip] = ['Medium', effective_faces_per_frame]
            medium += 1
        else:
            difficulty_dict[clip] = ['Hard', effective_faces_per_frame]
            hard += 1

    for idx, row in df_new.iterrows():
        df_new.at[idx, 'Difficulty'] = difficulty_dict[row['Clip_name']][0]
        df_new.at[idx, 'Faces_per_frame'] = difficulty_dict[row['Clip_name']][1]

    return df_new

def add_json_paths(df_new, part1_clips):
    # check each video clips
    for clip in part1_clips.keys():
        print(f'clip: {clip}')
        for row in df_new.iterrows():
            if clip not in row[1]['video_url']: continue
            df_new.at[row[0], 'Clip_name'] = clip
            df_new.at[row[0], 'Video_path'] = part1_clips[clip]['video']
            df_new.at[row[0], 'Appen_json'] = part1_clips[clip]['appen']
            df_new.at[row[0], 'Byte_json'] = part1_clips[clip]['byte']
            df_new.at[row[0], 'Ego4d_json'] = part1_clips[clip]['ego4d']
            df_new.at[row[0], 'Retina_0.3'] = part1_clips[clip]['retina_0.3']
            df_new.at[row[0], 'Retina_0.01'] = part1_clips[clip]['retina_0.01']
            df_new.at[row[0], 'Retina_0.8'] = part1_clips[clip]['retina_0.8']
    return df_new

def reject_outliers(df_new):
    # reject outlier jedgements in each video and group
    df_new['Outlier'] = 0
    unique_jobs = df_new['_unit_id'].unique()
    # unique_jobs = df_new['Clip_name'].unique()
    # outlier_label = 'Speed' # or 'Duration'
    outlier_label = 'Duration'
    for job in unique_jobs:
        # find upper and lower bound to reject outlier
        records = df_new[df_new['_unit_id'] == job][outlier_label]
        median = np.median(records)
        median_residuals = np.abs(records - median)
        MAD = np.median(median_residuals)  * 1.4826
        upper_bound = median + OUTLIER_FACTOR * MAD

        # lower_bound = max(0, median - 3 * MAD)
        lower_bound = 0.1
        # 0.1 * 60 = 6 mins, here we reject if someones spent less than 6 mins on a clip

        for row in df_new[df_new['_unit_id'] == job].iterrows():
            if row[1][outlier_label] > upper_bound or row[1][outlier_label] < lower_bound:
                df_new.at[row[0], 'Outlier'] = 1
    return df_new

def download_judgements(df, clips):
    api_key = "AXZngREC8oBJS-Hy14Wy"

    if len(clips) == 24:
        part_dir = BASE_DIR + 'part_1_merged_judgements/'
    elif len(clips) == 12:
        part_dir = BASE_DIR + 'part_2_merged_judgements/'

    # batch download reviewed judgements
    for i, row in df.iterrows():
        worker_id = row['_worker_id']
        group = row['Group']
        clip = row['Clip_name']

        clip_dir = os.path.join(part_dir, 'rater_judgements', clip)
        os.makedirs(clip_dir, exist_ok=True)

        out_path = os.path.join(clip_dir, f'{group}_{worker_id}_{clip}.json')

        if os.path.exists(out_path):
            df.loc[i, 'Rater_json'] = out_path
            continue

        annotation_pr = row['annotation_pr'].split('"')
        json_url = [s for s in annotation_pr if "https://" in s][0]
        json_url = json_url.replace("requestor-proxy", "api-beta")
        json_url += "&key=" + api_key

        with urllib.request.urlopen(json_url) as url:
            data = json.loads(url.read().decode())

            with open(out_path, 'w') as outfile:
                json.dump(data, outfile)
                # save judgement path to df
                df.loc[i, 'Rater_json'] = out_path
                print(f'{i} {clip} downloaded')

    return df


def match_id_and_tenure(df1, df2, id_tenure):
    # build three dictionaries for three types of worker IDs to match with
    # worker tenure in months
    # id1, id2, id3 = {}, {}, {}
    # df_both = pd.concat([df1, df2], ignore_index=True)
    id_dicts = {'id1':{}, 'id2':{}, 'id3':{}}
    groups = {'A': {}, 'B': {}, 'C': {}}
    missing_tenure = set()

    for row in id_tenure.iterrows():
        for id in id_dicts.keys():
            id_str = str(row[1][id])
            if id_str != 'N/A':
                tenure_month = int(row[1]['tenure_month'])
                id_dicts[id][id_str] = tenure_month
                groups[row[1]['Group']].setdefault(id_str, tenure_month)

    tenure_list = list(id_dicts['id1'].values())
    Novice_threshold = np.median(tenure_list)

    print(f'# of novice: {np.sum(np.array(tenure_list) < Novice_threshold)}')
    print(f'# of veterans: {np.sum(np.array(tenure_list) >= Novice_threshold)}')

    # plot the tenure distribution as a histogram using seaborn and save it
    sns.set(rc={'figure.figsize':(6, 4)})
    sns.histplot(tenure_list, bins=30)
    # draw a vertical line at the median tenure
    plt.axvline(x=Novice_threshold, color='r', linestyle='--')
    # set x y labels
    plt.xlabel('Experience (months)')
    plt.ylabel('Count')
    # make y axis only integer
    plt.yticks(np.arange(0, 16, 5))
    plt.tight_layout()
    plt.savefig(BASE_DIR + 'tenure_distribution.pdf')

    for df in [df1, df2]:

        # match worker ID and tenure
        for i, row in df.iterrows():
            worker_id = str(row['_worker_id'])
            if worker_id in id_dicts['id1'].keys():
                df.loc[i, 'tenure'] = id_dicts['id1'][worker_id]
            elif worker_id in id_dicts['id2'].keys():
                df.loc[i, 'tenure'] = id_dicts['id2'][worker_id]
            else:
                df.loc[i, 'tenure'] = -1

            curr_tenure = df.loc[i, 'tenure']

            if curr_tenure == -1:
                missing_tenure.add(worker_id)
                df.loc[i, 'tenure_group'] = 'Novice'
            elif curr_tenure <= Novice_threshold:
                df.loc[i, 'tenure_group'] = 'Novice'
            else:
                df.loc[i, 'tenure_group'] = 'Veteran'
        
    print(f'the following IDs are missing tenure: {missing_tenure}')

    return df1, df2

def n_way_label(df, column_label, labels):
    #  create a new column that contains the n-way label
    # for each row append the label
    df[column_label] = ''

    for i, row in df.iterrows():
        new_label = row[labels[0]]
        for label in labels[1:]:
            new_label = new_label + "_" + str(row[label])

        df.loc[i, column_label] = new_label

    return df

def df_preprocess(dfs, df2, part1_clips, part1_dir, part2_clips, part2_dir, id_tenure):
    groups = ['A', 'B', 'C']
    df_new = pd.DataFrame()
    df2_new = pd.DataFrame()

    # add group label and drop labels not used
    for i in range(len(groups)):
        temp = dfs[i][['_unit_id', '_created_at', '_id', '_started_at', '_worker_id', 'annotation_pr', 'video_url']]
        temp.insert(0, 'Group', groups[i])
        # df_new = df_new.append(temp, ignore_index=True)
        df_new = pd.concat([df_new, temp], ignore_index=True)

    # find out the right group for part 2 jobs
    unique_workers = df_new['_worker_id'].unique()

    for worker in unique_workers:
        temp = df2[['_unit_id', '_created_at', '_id', '_started_at', '_worker_id', 'annotation_pr', 'video_url']]
        temp_w = temp[temp['_worker_id'] == worker]

        for i in range(3):
            candidates = dfs[i][dfs[i]['_worker_id'] == worker]
            if len(candidates) > 10:
                temp_w['Group'] = groups[i]
                # df2_new = df2_new.append(temp_w, ignore_index=True)
                df2_new = pd.concat([df2_new, temp_w], ignore_index=True)

    for g in groups:
        print(f'Part 2, Group {g}:', len(df2_new[df2_new['Group']==g]['_worker_id'].unique()))

    df_new = add_time_and_speed(df_new)
    df2_new = add_time_and_speed(df2_new)

    df_new = add_json_paths(df_new, part1_clips)
    df2_new = add_json_paths(df2_new, part2_clips)

    df_new = add_difficulty(df_new)
    df2_new = add_difficulty(df2_new, part2=True)

    df_new = reject_outliers(df_new)
    df2_new = reject_outliers(df2_new)

    df_new = download_judgements(df_new, part1_clips)
    df2_new = download_judgements(df2_new, part2_clips)

    df_new, df2_new = match_id_and_tenure(df_new, df2_new, id_tenure)
    # df2_new = match_id_and_tenure(, id_tenure)

    df_new = df_new.sort_values(by=['Start_time'])
    df2_new = df2_new.sort_values(by=['Start_time'])

    # df_new = n_way_label(df_new, 'group_difficulty', ['Group', 'Difficulty'])
    # df_new = n_way_label(df_new, 'group_tenure', ['Group', 'tenure_group'])
    # df_new = n_way_label(df_new, 'group_difficulty_tenure', ['Group', 'Difficulty', 'tenure_group'])

    # df2_new = n_way_label(df2_new, 'group_difficulty', ['Group', 'Difficulty'])
    # df2_new = n_way_label(df2_new, 'group_tenure', ['Group', 'tenure_group'])
    # df2_new = n_way_label(df2_new, 'group_difficulty_tenure', ['Group', 'Difficulty', 'tenure_group'])
    
    # save new df to a csv file
    df_new.to_csv(os.path.join(part1_dir, 'df_combined.csv'), index=False)
    df2_new.to_csv(os.path.join(part2_dir, 'df2_combined.csv'), index=False)

    return df_new, df2_new

def boxplot(df, out_dir, order):
    fig_name = 'fig_boxplot_difficulty_speed_clean.pdf'
    
    # fig, ax = plt.subplots(figsize=(16,12))
    sns.set(rc={'figure.figsize':(12,6)})
    sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
    ax = sns.boxplot(x="Difficulty", y="Speed", hue='Group', data=df, order=order, linewidth=1.0)
    sns.swarmplot(x="Difficulty", y="Speed", hue='Group', dodge=True, data=df, order=order, size=2, palette="pastel")
    ax.set_ylabel('Speed (frames/hour)')
    ax.set_xlabel('Video difficulty')
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fig_name))

def barplot(df, out_dir, order, ci='se', ext=''):
    if ci == 'se':
        label = 'standard_error'
        ci_range = 68
    elif ci == 'sd':
        label = 'standard_deviation'
        ci_range = 95

    fig_name = f'fig_barplot_difficulty_speed_{label}_{ext}.pdf'

    # ci = 68% is one standard error
    ax = sns.catplot(x="Difficulty", y="Speed", hue="Group", kind="bar", ci=ci_range, data=df, order=order, aspect=2, legend=False)
    ax.set_ylabels('Speed (frames/hour)')
    ax.set_xlabels('Video difficulty')
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, fig_name))


def histogram_plots(df, out_dir, order):
    fig1_name = 'fig_histogram_matrix_speed.pdf'
    fig3_name = 'fig_histogram_stack_speed.pdf'

    # speed histogram per Difficulty level per group
    sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
    g = sns.FacetGrid(df, col="Difficulty", row='Group', hue='Group', col_order=order, height=3, aspect=2.0)
    # g.map(sns.histplot, "Speed", stat="probability", kde=True, binwidth=1.0, binrange=(0, 20))
    g.map(sns.histplot, "Speed", stat="probability", kde=True, binwidth=200.0, binrange=(0, 9000))
    g.refline(x=df["Speed"].median())
    g.set_axis_labels("Speed (frames/hour)", "Probability density")
    g.add_legend()
    plt.savefig(os.path.join(out_dir, fig1_name))
    plt.clf() # this clears the figure

    # speed histogram per Difficulty level per group
    # f, axs = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw=dict(width_ratios=[4, 3]))
    sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
    g = sns.FacetGrid(df, col="Difficulty", hue='Group', col_order=order, height=3, aspect=1.2)
    # g.map(sns.histplot, "Speed", kde=True, stat="probability", binwidth=1.0, binrange=(0, 20))
    g.map(sns.histplot, "Speed", stat="probability", kde=True, binwidth=200.0, binrange=(0, 9000))
    g.refline(x=df["Speed"].median())
    g.set_axis_labels("Speed (frames/hour)", "Probability density")
    g.add_legend()
    plt.savefig(os.path.join(out_dir, fig3_name))
    plt.clf() # this clears the figure


def pointplot(df, base_dir, order, part_label, label='Speed', ext=''):
    ylabel = 'Speed (frames/hour)' if label == 'Speed' else 'Task Time (hour)'
    title = f'{part_label} Average speed over difficulty' if label == 'Speed' else f'{part_label} average task time over video difficulty'
    if len(ext) > 0: title = f'{title} ({ext})'
    hue_order = ['A', 'B', 'C']
    
    # add a new Overall to Difficulty that includes all judgements
    df_copy = df.copy()
    df_copy['Difficulty'] = 'Overall'
    df_overall = pd.concat([df, df_copy])

    df_temp = df_overall[df_overall['Difficulty'] == 'Overall']

    # plt.figure()
    sns.set(rc={'figure.figsize':(6, 5)})
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, sharey=True, gridspec_kw={'width_ratios': [1, 3, 2]})

    # left figure, Overvall 
    sns.pointplot(ax=ax1, data=df_temp, x='Difficulty', order=order[:1], hue_order=hue_order, y=label, hue='Group', dodge=0.3, capsize=.1, ci=95, seed=0, join=False)
    ax1.set(xticklabels=[])  
    ax1.set(xlabel='Group', ylabel=ylabel) #, title=title)
    ax1.get_legend().remove()

    # right figure, per Difficulty
    sns.pointplot(ax=ax2, data=df, x='Difficulty', order=order[1:], hue_order=hue_order, y=label, hue='Group', dodge=0.3, capsize=.1, ci=95, seed=0, join=True)
    ax2.set(xlabel='Video Difficulty', ylabel=None) #, title=title)
    ax2.get_legend().remove()
    # plt.legend(loc='lower right')

    sns.pointplot(ax=ax3, data=df, x='tenure_group', order=['Novice', 'Veteran'], hue_order=hue_order, y=label, hue='Group', dodge=0.3, capsize=.1, ci=95, seed=0, join=True)
    ax3.set(xlabel='User Experience', ylabel=None)
    ax3.get_legend().remove()

    plt.ylim(0.0, 1.45)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, f'fig_{part_label}_task_time.pdf'))
    plt.savefig(os.path.join(base_dir, f'fig_{part_label}_task_time.png'))

    # collect data for print
    for group in ['A', 'B', 'C']:
        for difficulty in order:
            mean = np.mean(df[(df['Difficulty'] == difficulty) & (df['Group'] == group)][label])
            print(f'{part_label} {group} {difficulty} mean: {mean:.2f}')

    plt.clf() # this clears the figure
    print(f'Plotting done.')

def statistical_analysis(df, part='part_1'):

    ### two-way ANOVA analysis
    # ANOVA table using statsmodels
    model = ols('Duration ~ C(Group) + C(Difficulty) + C(Group):C(Difficulty)', data=df).fit()
    #  set `typ=3`. Type 3 sums of squares (SS) does not assume equal sample sizes among the groups
    #  and is recommended for an unbalanced design for multifactorial ANOVA.
    anova_table = sm.stats.anova_lm(model, typ=3)
    print("two-way anova unbalanced sample size\n", anova_table)
    print(f'Done {part} two-way ANOVA.\n')

    # Part 1
    #                              sum_sq      df           F        PR(>F)
    # Intercept                47.590102     1.0  166.857381  2.205071e-36
    # C(Group)                  0.379155     2.0    0.664684  5.145779e-01
    # C(Difficulty)            36.664425     2.0   64.275235  1.411571e-27
    # C(Group):C(Difficulty)    7.251887     4.0    6.356526  4.497477e-05
    # Residual                456.628004  1601.0         NaN           NaN

    # Part 2
    #                              sum_sq     df          F        PR(>F)
    # Intercept                22.030545    1.0  74.007473  4.299310e-17
    # C(Group)                  0.885155    2.0   1.486756  2.267500e-01
    # C(Difficulty)             9.491163    2.0  15.941889  1.641828e-07
    # C(Group):C(Difficulty)    0.605110    4.0   0.508188  7.297425e-01
    # Residual                230.404320  774.0        NaN           NaN

    breakpoint()

    # ANOVA table using bioinfokit v1.0.3 or later (it uses wrapper script for anova_lm)
    # from bioinfokit.analys import stat
    # res = stat()
    # res.anova_stat(df=df, res_var='Duration', anova_model='Duration~C(Group)+C(Difficulty)+C(Group):C(Difficulty)')
    # res.anova_summary
    """
    Part 1
                                df      sum_sq    mean_sq          F        PR(>F)
    C(Group)                   2.0   25.303086  12.651543  44.358034  1.779783e-19
    C(Difficulty)              2.0   50.718416  25.359208  88.912838  2.418149e-37
    C(Group):C(Difficulty)     4.0    7.251887   1.812972   6.356526  4.497477e-05
    Residual                1601.0  456.628004   0.285214        NaN           NaN
    """

    breakpoint()

    # Power of the ANOVA    
    k = df1_anova['Group'].nunique()  # Number of groups
    n = df1_anova.shape[0] / k  # Number of observations per group
    achieved_power = pg.power_anova(eta_squared=aov.loc[0, 'np2'], k=k, n=n, alpha=0.05)
    print('Achieved power: %.4f' % achieved_power)
    """
    0.999998940677179
    Achieved power: 1.0000
    """

    breakpoint()

    ### two-way ANOVA analysis
    pg.mixed_anova(data=df1_anova, dv='Speed', between='Group', within='Difficulty', subject='Clip_name')

    ### Repeated measures ANOVA, which requires within subject factors, not sure if it is suitable
    # res_one_way = pg.rm_anova(data=df1_anova, dv='Speed', within='Difficulty', subject='_id', detailed=True)
    # print("one-way repeated measures ANOVA\n", res_one_way)

    # res_two_way = pg.rm_anova(data=df1_anova, dv='Speed', within=['Group', 'Difficulty'])
    # print("two-way repeated measures ANOVA\n", res_two_way)

def two_way_anova(df, part='part_1'):
    # N-way ANOVA using Type 3 sum of squares (SS) assumption to address unequal sample sizes
    aov = pg.anova(data=df, dv='Duration', between=['Group', 'Difficulty'], ss_type=3, detailed=True)
    print(f'{part} two-way ANOVA\n', aov)

    """
                   Source          SS      DF         MS          F         p-unc       np2
    0               Group   20.120936     2.0  10.060468  35.273371  1.020661e-15  0.042204
    1          Difficulty   50.744054     2.0  25.372027  88.957783  2.322283e-37  0.100013
    2  Group * Difficulty    7.251887     4.0   1.812972   6.356526  4.497477e-05  0.015633
    3            Residual  456.628004  1601.0   0.285214        NaN           NaN       NaN
    """

    breakpoint()


def anova_or_welch(df, part='part_1'):
    ### One-way Welch's ANOVA analysis

    # 1. This is a between subject design, so the first step is to test for equality of variances
    print(pg.homoscedasticity(data=df, dv='Duration', group='Group'))
    """
    Part 1 overall --> Welch's t-test
                    W          pval  equal_var
    levene  16.171457  1.113182e-07      False
    
    part 1 Easy --> Anova
                   W      pval  equal_var
    levene  0.167663  0.845697       True

    Part 1 Medium --> Welch's t-test
                    W      pval  equal_var
    levene  10.809436  0.000024      False

    Part 1 Hard --> Welch's t-test
                   W      pval  equal_var
    levene  3.496587  0.030926      False

    Part 2 everything --> Anova
                   W      pval  equal_var
    levene  1.973515  0.139661       True
    """

    if part == 'part_2':
        aov = pg.anova(data=df, dv='Duration', between='Group', detailed=True)
        print(f'{part} one-way ANOVA\n', aov)
        """
        Part 1 Easy
           Source         SS   DF        MS         F     p-unc       np2
        0   Group   0.379155    2  0.189577  1.237938  0.291079  0.006091
        1  Within  61.868422  404  0.153140       NaN       NaN       NaN

        Part 2 overall
           Source          SS   DF        MS        F     p-unc       np2
        0   Group    4.491361    2  2.245680  6.68858  0.001318  0.016861
        1  Within  261.883807  780  0.335748      NaN       NaN       NaN
        
        Easy
        0   Group   0.885155    2  0.442577  2.016092  0.136014  0.020781
        1  Within  41.709269  190  0.219522       NaN       NaN       NaN

        Medium
        0   Group   0.619221    2  0.309610  1.541898  0.215963  0.011949
        1  Within  51.203535  255  0.200798       NaN       NaN       NaN

        Hard
            Source          SS   DF        MS         F     p-unc      np2
        0   Group    3.302970    2  1.651485  3.951797  0.020139  0.02346
        1  Within  137.491516  329  0.417907       NaN       NaN      NaN
        """

        # post-hoc test for one-way ANOVA
        tukey = pg.pairwise_tukey(data=df, dv='Duration', between='Group')
        print(f'{part} one-way ANOVA post-hoc test\n', tukey)
        """
        Part 2 Overall
           A  B   mean(A)   mean(B)      diff        se         T   p-tukey    hedges
        0  A  B  0.865247  1.021978 -0.156731  0.050261 -3.118337  0.005351 -0.270105
        1  A  C  0.865247  0.860869  0.004379  0.051322  0.085315  0.995995  0.007545
        2  B  C  1.021978  0.860869  0.161109  0.050670  3.179557  0.004369  0.277645

        Easy
           A  B   mean(A)   mean(B)      diff        se         T   p-tukey    hedges
        0  A  B  0.596097  0.760227 -0.164130  0.082866 -1.980672  0.119770 -0.348218
        1  A  C  0.596097  0.657154 -0.061057  0.083174 -0.734086  0.743564 -0.129532
        2  B  C  0.760227  0.657154  0.103073  0.081874  1.258926  0.420228  0.218711

        Medium
           A  B   mean(A)   mean(B)      diff        se         T   p-tukey    hedges
        0  A  B  0.798241  0.877989 -0.079747  0.067559 -1.180415  0.465943 -0.177198
        1  A  C  0.798241  0.760671  0.037571  0.068969  0.544743  0.849254  0.083466
        2  B  C  0.877989  0.760671  0.117318  0.068592  1.710365  0.203259  0.260646

        Hard
           A  B   mean(A)   mean(B)      diff        se         T   p-tukey    hedges
        0  A  B  1.069945  1.276983 -0.207038  0.085678 -2.416457  0.042739 -0.319201
        1  A  C  1.069945  1.067192  0.002753  0.088417  0.031138  0.999466  0.004244
        2  B  C  1.276983  1.067192  0.209791  0.086948  2.412834  0.043136  0.323416
        """

    else:
        # 2. equal variance not met, so we use a welch t-test
        # https://pingouin-stats.org/generated/pingouin.welch_anova.html#pingouin.welch_anova
        aov_welch = pg.welch_anova(data=df, dv='Duration', between='Group')
        print("Welch's t-test\n", aov_welch)

        """
        Part 1 Overall
          Source  ddof1        ddof2          F         p-unc       np2
        0  Group      2  1058.034423  46.760886  3.466752e-20  0.049296

        Part 1 Medium
          Source  ddof1       ddof2          F         p-unc       np2
        0  Group      2  387.722367  41.480419  4.735514e-17  0.103722

        Part 1 Hard
          Source  ddof1       ddof2          F         p-unc       np2
        0  Group      2  395.264186  18.666171  1.793488e-08  0.053702

        'p-unc': uncorrected p-values
        F-value = variation between sample means / variation within the samples

        df1 = df of the explained part = number of groups — 1
        df2 = df of the residual = number of observations — number of groups
        Degrees of freedom 2 (denominator) here = 1058? But 1610 rows in the dataframe.

        https://www.statology.org/what-does-a-high-f-value-mean/
        
        """

        # 3. if the groups have unequal variances, the Games-Howell test is the more adequate post-hoc test
        pg.pairwise_gameshowell(data=df, dv='Duration', between='Group')
        """
        Part 1 Overall
        A  B   mean(A)   mean(B)      diff        se         T           df          pval    hedges
        0  A  B  1.044505  0.910953  0.133552  0.036936  3.615803  1082.994206  9.132381e-04  0.219425
        1  A  C  1.044505  0.730410  0.314096  0.033243  9.448595  1021.748399  3.391731e-13  0.575603
        2  B  C  0.910953  0.730410  0.180544  0.032929  5.482883  1002.302774  1.586628e-07  0.336899

        Part 1 Medium
           A  B   mean(A)   mean(B)      diff        se         T          df      pval    hedges
        0  A  B  1.161854  0.956636  0.205217  0.057371  3.577022  407.788511  0.001131  0.352869
        1  A  C  1.161854  0.727923  0.433930  0.048910  8.872086  361.083885  0.000000  0.877519
        2  B  C  0.956636  0.727923  0.228713  0.048519  4.713854  343.239865  0.000011  0.474058

        Part 1 Hard
           A  B   mean(A)   mean(B)      diff        se         T          df      pval    hedges
        0  A  B  1.220000  1.047525  0.172475  0.064421  2.677309  402.361943  0.021047  0.264934
        1  A  C  1.220000  0.856947  0.363053  0.059523  6.099338  392.132328  0.000000  0.613052
        2  B  C  1.047525  0.856947  0.190577  0.062035  3.072107  383.805917  0.006429  0.309878
        """
    breakpoint()


if __name__ == "__main__":
    prefix = "AB testing time analysis"
    api_key = "AXZngREC8oBJS-Hy14Wy"

    id_tenure_csv = "user_study_results/id_tenure_pairs.csv"

    part1_dir = 'user_study_results/part_1_progress_check/'
    part2_dir = 'user_study_results/part_2_progress_check/'

    part1_raw = 'data/grouped_tasks'
    part2_raw = 'data/shared_tasks'

    # Part 1 clips
    part1_clips = collect_AB_clips(part1_raw)
    part2_clips = collect_AB_clips(part2_raw)

    # group CSVs
    group_A_csv = part1_dir + 'group_a_part_1_99.8.csv'
    group_B_csv = part1_dir + 'group_b_part_1_99.6.csv'
    group_C_csv = part1_dir + 'group_c_part_1_99.5.csv'
    part2_csv = part2_dir + '98_percent_926_judgements.csv' 

    df_A = pd.read_csv(group_A_csv)
    df_B = pd.read_csv(group_B_csv)
    df_C = pd.read_csv(group_C_csv)
    part2_df = pd.read_csv(part2_csv)
    id_tenure = pd.read_csv(id_tenure_csv, dtype=str)

    part1_dfs = [df_A, df_B, df_C]
    groups = ["group_A", "group_B", "group_C"]

    df1, df2 = df_preprocess(part1_dfs, part2_df, part1_clips, part1_dir, part2_clips, part2_dir, id_tenure)

    df1_anova = df1[df1['Outlier'] == 0]
    df2_anova = df2[df2['Outlier'] == 0]

    order = ["Overall", "Easy", "Medium", "Hard"]

    # pointplot(df1_anova, part1_dir, order, 'Part 1', 'Speed')
    pointplot(df1_anova, part1_dir, order, 'part1', 'Duration', ext='ext')
    # Part 2 clips
    pointplot(df2_anova, part2_dir, order, 'part2', 'Duration', ext='Without AI assistance')

    #### Paper submission ANOVA/Welch Analysis

    # given different video difficulty
    # df1_temp = df1_anova[(df1_anova['Difficulty'] == 'Easy')]
    # anova_or_welch(df1_temp, 'part_1')

    # df2_temp = df2_anova[df2_anova['Difficulty'] == 'Easy']
    # anova_or_welch(df2_temp, 'part_2')


    #### Revision -- redo N-way ANOVA

    df1_mini = df1_anova[['Group', 'Duration', 'Difficulty' ,'tenure_group']]
    import scipy.stats as stats
    from sklearn.preprocessing import PowerTransformer
    
    # boxcox transform
    duration = np.array(df1_mini['Duration'])
    # df1_mini['Duration_transformed'] = stats.boxcox(duration, lmbda=0.1)

    breakpoint()
    # Power transform
    # pt = PowerTransformer(method='box-cox', standardize=True)
    # duration = duration.reshape(-1, 1)
    # print(pt.fit(duration))
    # print(pt.transform(duration))

    # df1_mini['Duration_transformed'] = pt.transform(duration).flatten()

    # print(df1_mini['Duration_transformed'].describe())
    # count    1.610000e+03
    # mean    -6.619963e-18
    # std      1.000311e+00
    # min     -2.621209e+00
    # 25%     -6.748357e-01
    # 50%      2.594146e-02
    # 75%      6.732021e-01
    # max      3.012433e+00

    print(pg.homoscedasticity(df1_mini, dv='Duration_transformed', group='Group', method='levene'))
    # (Pdb)     print(pg.homoscedasticity(df1_mini, dv='Duration_transformed', group='Group', method='levene'))
    #                W     pval  equal_var
    # levene  1.370171  0.25436       True

    # calculate each group's variance
    # print(f'Outlier upper bound: mean + {OUTLIER_FACTOR} * std')
    # for g in df1.Group.unique():
    #     for d in df1.Difficulty.unique():
    #         condition = (df1['Group'] == g) & (df1['Difficulty'] == d)
    #         print(f'Group {g}, Difficulty {d}', )
    #         print(f'{len(df1[condition])} After outlier rejection', len(df1_anova[condition]))
    #         print(f'mean: {df1[condition].Duration.mean().round(2)}')
    #         print(f'variance: {df1[condition].Duration.var().round(2)}')

    # print(f"Group mean: {df1_anova.groupby('Group')['Duration'].mean()}")
    # print(f"Group variance: {df1_anova.groupby('Group')['Duration'].var()}")

    print(pg.homoscedasticity(data=df1_mini, dv='Duration', group='Group'))
    #                 W          pval  equal_var
    # levene  16.171457  1.113182e-07      False

    print(pg.homoscedasticity(data=df1_mini, dv='Recall', group='Group'))


    # Type II Sums of Squares should be used if there is no interaction between the independent variables.
    # Unlike Type II, the Type III Sums of Squares do specify an interaction effect.
    # https://towardsdatascience.com/anovas-three-types-of-estimating-sums-of-squares-don-t-make-the-wrong-choice-91107c77a27a

    between = ['Group', 'Difficulty', 'tenure_group']

    print(pg.anova(data=df1_mini, dv='Duration', between=between, detailed=True, effsize='np2', ss_type=3).round(4))
    #                               Source        SS      DF       MS        F   p-unc     np2
    # 0                              Group   17.7874     2.0   8.8937  32.0112  0.0000  0.0387
    # 1                         Difficulty   50.0468     2.0  25.0234  90.0669  0.0000  0.1016
    # 2                       tenure_group    9.9911     1.0   9.9911  35.9611  0.0000  0.0221
    # 3                 Group * Difficulty    7.1986     4.0   1.7997   6.4775  0.0000  0.0160
    # 4               Group * tenure_group    0.9031     2.0   0.4515   1.6252  0.1972  0.0020
    # 5          Difficulty * tenure_group    1.2041     2.0   0.6021   2.1670  0.1149  0.0027
    # 6  Group * Difficulty * tenure_group    1.0789     4.0   0.2697   0.9708  0.4224  0.0024
    # 7                           Residual  442.3070  1592.0   0.2778      NaN     NaN     NaN

    print(pg.pairwise_tukey(data=df1_mini, dv='Duration', between='Group', padjust='holm').round(4))

    print(pg.pairwise_gameshowell(data=df1_anova, dv='Duration', between='group_difficulty', effsize='eta-square').round(4))
    #            A         B  mean(A)  mean(B)    diff      se        T        df    pval  eta-square
    # 0     A_Easy    A_Hard   0.5937   1.2200 -0.6263  0.0546 -11.4753  335.5394  0.0000      0.2880
    # 1     A_Easy  A_Medium   0.5937   1.1619 -0.5681  0.0522 -10.8917  344.9999  0.0000      0.2645
    # 2     A_Easy    B_Easy   0.5937   0.6355 -0.0418  0.0482  -0.8663  263.3828  0.9945      0.0028
    # 3     A_Easy    B_Hard   0.5937   1.0475 -0.4538  0.0573  -7.9191  326.6341  0.0000      0.1623
    # 4     A_Easy  B_Medium   0.5937   0.9566 -0.3629  0.0518  -7.0065  330.8280  0.0000      0.1326
    # 5     A_Easy    C_Easy   0.5937   0.5609  0.0328  0.0461   0.7116  271.9744  0.9986      0.0018
    # 6     A_Easy    C_Hard   0.5937   0.8569 -0.2632  0.0517  -5.0876  322.4512  0.0000      0.0758
    # 7     A_Easy  C_Medium   0.5937   0.7279 -0.1342  0.0422  -3.1775  288.2123  0.0429      0.0306
    # 8     A_Hard  A_Medium   1.2200   1.1619  0.0581  0.0599   0.9708  411.7596  0.9883      0.0023
    # 9     A_Hard    B_Easy   1.2200   0.6355  0.5845  0.0565  10.3438  335.9641  0.0000      0.2490
    # 10    A_Hard    B_Hard   1.2200   1.0475  0.1725  0.0644   2.6773  402.3619  0.1593      0.0173
    # 11    A_Hard  B_Medium   1.2200   0.9566  0.2634  0.0596   4.4208  399.2677  0.0004      0.0463
    # 12    A_Hard    C_Easy   1.2200   0.5609  0.6591  0.0547  12.0546  338.9351  0.0000      0.3049
    # 13    A_Hard    C_Hard   1.2200   0.8569  0.3631  0.0595   6.0993  392.1323  0.0000      0.0862
    # 14    A_Hard  C_Medium   1.2200   0.7279  0.4921  0.0515   9.5593  337.0991  0.0000      0.1857
    # 15  A_Medium    B_Easy   1.1619   0.6355  0.5263  0.0542   9.7150  339.6946  0.0000      0.2240
    # 16  A_Medium    B_Hard   1.1619   1.0475  0.1143  0.0624   1.8325  400.5662  0.6604      0.0081
    # 17  A_Medium  B_Medium   1.1619   0.9566  0.2052  0.0574   3.5770  407.7885  0.0116      0.0303
    # 18  A_Medium    C_Easy   1.1619   0.5609  0.6009  0.0523  11.4978  348.9650  0.0000      0.2825
    # 19  A_Medium    C_Hard   1.1619   0.8569  0.3049  0.0573   5.3196  399.3077  0.0000      0.0659
    # 20  A_Medium  C_Medium   1.1619   0.7279  0.4339  0.0489   8.8721  361.0839  0.0000      0.1619
    # 21    B_Easy    B_Hard   0.6355   1.0475 -0.4120  0.0591  -6.9660  331.4714  0.0000      0.1314
    # 22    B_Easy  B_Medium   0.6355   0.9566 -0.3211  0.0538  -5.9659  327.1056  0.0000      0.1006
    # 23    B_Easy    C_Easy   0.6355   0.5609  0.0746  0.0483   1.5425  266.8042  0.8343      0.0087
    # 24    B_Easy    C_Hard   0.6355   0.8569 -0.2214  0.0538  -4.1182  319.9474  0.0016      0.0514
    # 25    B_Easy  C_Medium   0.6355   0.7279 -0.0924  0.0447  -2.0673  267.1473  0.4980      0.0133
    # 26    B_Hard  B_Medium   1.0475   0.9566  0.0909  0.0621   1.4640  389.7068  0.8715      0.0053
    # 27    B_Hard    C_Easy   1.0475   0.5609  0.4866  0.0574   8.4774  329.5280  0.0000      0.1791
    # 28    B_Hard    C_Hard   1.0475   0.8569  0.1906  0.0620   3.0721  383.8059  0.0573      0.0235
    # 29    B_Hard  C_Medium   1.0475   0.7279  0.3196  0.0544   5.8793  318.6748  0.0000      0.0799
    # 30  B_Medium    C_Easy   0.9566   0.5609  0.3957  0.0519   7.6244  334.6505  0.0000      0.1511
    # 31  B_Medium    C_Hard   0.9566   0.8569  0.0997  0.0570   1.7494  385.8628  0.7153      0.0078
    # 32  B_Medium  C_Medium   0.9566   0.7279  0.2287  0.0485   4.7139  343.2399  0.0001      0.0534
    # 33    C_Easy    C_Hard   0.5609   0.8569 -0.2960  0.0518  -5.7099  326.1583  0.0000      0.0922
    # 34    C_Easy  C_Medium   0.5609   0.7279 -0.1670  0.0424  -3.9421  293.7269  0.0032      0.0456
    # 35    C_Hard  C_Medium   0.8569   0.7279  0.1290  0.0485   2.6627  331.9280  0.1655      0.0180

    print(pg.pairwise_gameshowell(data=df1_anova, dv='Duration', between='group_tenure', effsize='eta-square').round(4))
    #             A          B  mean(A)  mean(B)    diff      se        T        df    pval  eta-square
    # 0    A_Novice  A_Veteran   0.9301   1.1649 -0.2348  0.0520  -4.5143  515.8218  0.0001      0.0356  *** 
    # 1    A_Novice   B_Novice   0.9301   0.8598  0.0703  0.0445   1.5796  572.3614  0.6124      0.0043
    # 2    A_Novice  B_Veteran   0.9301   0.9739 -0.0438  0.0541  -0.8094  455.3274  0.9658      0.0013
    # 3    A_Novice   C_Novice   0.9301   0.6680  0.2621  0.0390   6.7153  507.0795  0.0000      0.0704  ***  Novice A/C
    # 4    A_Novice  C_Veteran   0.9301   0.8233  0.1068  0.0496   2.1539  450.4466  0.2618      0.0095
    # 5   A_Veteran   B_Novice   1.1649   0.8598  0.3051  0.0511   5.9738  508.1870  0.0000      0.0597  ***
    # 6   A_Veteran  B_Veteran   1.1649   0.9739  0.1910  0.0596   3.2019  498.4815  0.0181      0.0198  *    Veteran A/B
    # 7   A_Veteran   C_Novice   1.1649   0.6680  0.4969  0.0464  10.7145  416.4397  0.0000      0.1653  ***
    # 8   A_Veteran  C_Veteran   1.1649   0.8233  0.3415  0.0555   6.1505  477.1977  0.0000      0.0741  ***  Veteran A/C
    # 9    B_Novice  B_Veteran   0.8598   0.9739 -0.1141  0.0532  -2.1439  444.5896  0.2668      0.0086
    # 10   B_Novice   C_Novice   0.8598   0.6680  0.1918  0.0378   5.0776  538.5635  0.0000      0.0407  ***  Novice B/C
    # 11   B_Novice  C_Veteran   0.8598   0.8233  0.0364  0.0486   0.7502  442.0108  0.9754      0.0011  *
    # 12  B_Veteran   C_Novice   0.9739   0.6680  0.3059  0.0487   6.2763  357.7554  0.0000      0.0677
    # 13  B_Veteran  C_Veteran   0.9739   0.8233  0.1506  0.0575   2.6175  445.0889  0.0951      0.0151  *    Veteran B/C
    # 14   C_Novice  C_Veteran   0.6680   0.8233 -0.1554  0.0436  -3.5624  350.6680  0.0056      0.0245


    # CORRECTION: Difficulty is not within-subjects, because they are different videos?

    print(pg.mixed_anova(data=df1_anova, dv='Duration', within='Difficulty', between='Group', subject='_worker_id').round(4))
    #         Source       SS  DF1  DF2      MS        F   p-unc     np2     eps
    # 0        Group   2.9605    2   73  1.4802   4.2155  0.0185  0.1035     NaN
    # 1   Difficulty  10.3246    2  146  5.1623  87.8181  0.0000  0.5461  0.9568
    # 2  Interaction   0.8031    4  146  0.2008   3.4155  0.0106  0.0856     NaN

    print(pg.mixed_anova(data=df1_anova, dv='Duration', within='tenure_group', between='Group', subject='_worker_id').round(4))
    # *** ValueError: cannot convert float NaN to integer

    print(pg.pairwise_tests(data=df1_anova, dv='Duration', within='Difficulty', between='Group', subject='_worker_id', padjust='holm', effsize='eta-square', interaction=True, correction='auto').round(4))
    #               Contrast Difficulty     A       B Paired  Parametric        T      dof alternative   p-unc  p-corr p-adjust       BF10  eta-square
    # 0           Difficulty          -  Easy    Hard   True        True -11.6550  75.0000   two-sided  0.0000  0.0000     holm  3.292e+15      0.2687
    # 1           Difficulty          -  Easy  Medium   True        True -10.3488  75.0000   two-sided  0.0000  0.0000     holm  1.504e+13      0.2159
    # 2           Difficulty          -  Hard  Medium   True        True   3.0484  75.0000   two-sided  0.0032  0.0032     holm      8.745      0.0195
    # 3                Group          -     A       B  False        True   0.8866  50.0000   two-sided  0.3795  0.3795     holm      0.385      0.0149
    # 4                Group          -     A       C  False        True   2.9525  47.9836   two-sided  0.0049  0.0146     holm      8.549      0.1476
    # 5                Group          -     B       C  False        True   1.9782  47.8398   two-sided  0.0537  0.1074     holm      1.366      0.0719
    # 6   Difficulty * Group       Easy     A       B  False        True  -0.1631  50.0000   two-sided  0.8711  0.9998     holm      0.281      0.0005
    # 7   Difficulty * Group       Easy     A       C  False        True   1.0097  47.9387   two-sided  0.3177  0.9754     holm      0.429      0.0198
    # 8   Difficulty * Group       Easy     B       C  False        True   1.1799  47.9443   two-sided  0.2438  0.9754     holm        0.5      0.0269
    # 9   Difficulty * Group       Hard     A       B  False        True   0.6796  50.0000   two-sided  0.4999  0.9998     holm      0.337      0.0088
    # 10  Difficulty * Group       Hard     A       C  False        True   2.4015  47.6938   two-sided  0.0203  0.1419     holm      2.809      0.1036
    # 11  Difficulty * Group       Hard     B       C  False        True   1.4696  47.1431   two-sided  0.1483  0.7415     holm      0.681      0.0408
    # 12  Difficulty * Group     Medium     A       B  False        True   1.6101  50.0000   two-sided  0.1137  0.6820     holm      0.801      0.0475
    # 13  Difficulty * Group     Medium     A       C  False        True   4.4485  44.9824   two-sided  0.0001  0.0005     holm    393.247      0.2783
    # 14  Difficulty * Group     Medium     B       C  False        True   2.6512  45.2725   two-sided  0.0110  0.0881     holm      4.544      0.1206


    print(pg.pairwise_tests(data=df1_anova, dv='Duration', within='Difficulty', between='tenure_group', subject='_worker_id', padjust='holm', effsize='eta-square', interaction=True, correction='auto').round(4))

#                     Contrast Difficulty       A        B Paired  Parametric        T      dof alternative   p-unc  p-corr p-adjust       BF10  eta-square
# 0                 Difficulty          -    Easy     Hard   True        True -11.7054  75.0000   two-sided  0.0000  0.0000     holm  4.039e+15      0.2715
# 1                 Difficulty          -    Easy   Medium   True        True -10.3505  75.0000   two-sided  0.0000  0.0000     holm  1.514e+13      0.2160
# 2                 Difficulty          -    Hard   Medium   True        True   3.0992  75.0000   two-sided  0.0027  0.0027     holm     10.002      0.0203
# 3               tenure_group          -  Novice  Veteran  False        True  -2.5851  62.8444   two-sided  0.0121     NaN      NaN       3.99      0.0845
# 4  Difficulty * tenure_group       Easy  Novice  Veteran  False        True  -2.2519  63.0809   two-sided  0.0278  0.0556     holm       2.05      0.0654
# 5  Difficulty * tenure_group       Hard  Novice  Veteran  False        True  -2.6781  63.1132   two-sided  0.0094  0.0283     holm      4.876      0.0901
# 6  Difficulty * tenure_group     Medium  Novice  Veteran  False        True  -1.7728  58.6154   two-sided  0.0815  0.0815     holm      0.914      0.0422


    print(pg.mixed_anova(data=df1_anova, dv='Duration', within='tenure_group', between='Group', subject='_worker_id').round(5))
    # *** ValueError: cannot convert float NaN to integer

    # post hoc tests
    print(pg.pairwise_tukey(data=df1_anova, dv='Duration', between=['Group', 'Difficulty'], effsize='eta-square').round(4))
    # *** IndexError: arrays used as indices must be of integer (or boolean) type


    # Part2
    ###########################

    print(pg.anova(data=df2_anova, dv='Duration', between=between, detailed=True, effsize='np2', ss_type=3).round(3))
    #                               Source       SS     DF      MS       F  p-unc    np2
    # 0                              Group    3.015    2.0   1.507   5.266  0.005  0.014
    # 1                         Difficulty   30.010    2.0  15.005  52.418  0.000  0.121
    # 2                       tenure_group    6.965    1.0   6.965  24.331  0.000  0.031
    # 3                 Group * Difficulty    0.455    4.0   0.114   0.397  0.811  0.002
    # 4               Group * tenure_group    2.776    2.0   1.388   4.848  0.008  0.013
    # 5          Difficulty * tenure_group    0.180    2.0   0.090   0.315  0.730  0.001
    # 6  Group * Difficulty * tenure_group    1.635    4.0   0.409   1.428  0.223  0.007
    # 7                           Residual  218.985  765.0   0.286     NaN    NaN    NaN
