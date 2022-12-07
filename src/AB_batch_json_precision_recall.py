"""This script takes many ADAP format bounding box JSON files 
to evaluate the detections in precision and recall.
"""
import os
import json
import glob
import numpy as np
import pandas as pd
from ADAP_json_utils import precision_recall_eval

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pingouin as pg

from sklearn.preprocessing import PowerTransformer


def one_video_precision_recall(gt_json, rater_judgements, cost_thresholds, consider_optional):
    # thresholds used in linear_assignment, thus
    # IOU_thres = 1-thresholds
    # steps = 9
    # labels = [str(t) for t in thresholds]
    with open(gt_json, 'r') as f:
        gt = json.load(f)

    # load and analyze all judgements from participants
    frames_total = len(gt['annotation']['frames'])
    all_precisions = []
    all_recalls = []

    for i in range(len(rater_judgements)):
        p  = rater_judgements[i]
        with open(p, 'r') as f:
            target = json.load(f)

        precisions, recalls = precision_recall_eval(gt, target, cost_thresholds, consider_optional)
        all_precisions.append(precisions)
        all_recalls.append(recalls)

    # average over all judgements
    mean_precisions = np.mean(all_precisions, axis=0)
    mean_recalls = np.mean(all_recalls, axis=0)

    return mean_precisions, mean_recalls

def scatterplot_video(df_pr, order, cost_thresholds):
    # scatter plot for each video
    sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})

    g = sns.FacetGrid(df_pr, col="difficulty", col_order=order, hue="method")
    g.map(sns.scatterplot, "recall", "precision")
    g.set_axis_labels("Recall", "Precision")

    plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0)
    figure_name = f'AB_testing_precision_recall_IOU_{1-cost_thresholds[0]}.png'
    figure_title = "AB Testing Preicision-Recall"
    plt.tight_layout()
    plt.savefig(judge_dir+figure_name)
    plt.clf() # this clears the figure


def scatterplot_method(df_pr, order, cost_thresholds):
    # scatter plot for each method (average over videos)
    sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
    # average over videos
    df_mean = df_pr.groupby(['method', 'difficulty', 'threshold']).mean()
    df_mean = df_mean.reset_index()
    g = sns.FacetGrid(df_mean, col="difficulty", col_order=order, hue="method")
    # g = sns.FacetGrid(df_mean, col="difficulty", col_order=order, hue="method", xlim=(0,1), ylim=(0,100))
    g.map(sns.scatterplot, "recall", "precision")
    g.set_axis_labels("Recall", "Precision")
    # plt.xlim(90,100)
    # plt.ylim(90,100)
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', borderaxespad=0)
    plt.tight_layout()
    plt.savefig(judge_dir+f'AB_testing_precision_recall_method_mean_IOU_{1-cost_thresholds[0]}.png')

def improvement_plot_single(df_pr, PR_F1, judge_dir, part2=False):
    # plot the transition from AI pre-annotation to rater judgements
    sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
    plt.figure(figsize=(7, 4))
    # save precision recall as x, y coordinates for arrow plotting
    prefix = 'Part2' if part2 else 'Part1'

    # read emoji images as markers
    zm_s = 0.05
    zm_l = 0.06
    robot_green = OffsetImage(plt.imread('evaluation/robot_1f916_green.png'), zoom=zm_l)
    robot_orange = OffsetImage(plt.imread('evaluation/robot_1f916_orange.png'), zoom=zm_l)
    robot_grey = OffsetImage(plt.imread('evaluation/robot_1f916_grey.png'), zoom=zm_l)
    human_green = OffsetImage(plt.imread('evaluation/busts-in-silhouette_1f465_green.png'), zoom=zm_s)
    human_orange = OffsetImage(plt.imread('evaluation/busts-in-silhouette_1f465_orange.png'), zoom=zm_s)
    human_blue = OffsetImage(plt.imread('evaluation/busts-in-silhouette_1f465_blue.png'), zoom=zm_s)
    emojis = [human_blue, robot_grey, robot_orange, human_orange, robot_green, human_green]

    # plot the points
    condition = (df_pr['Difficulty'] == 'Overall')
    df_difficulty = df_pr.loc[condition]

    pres, recs = [], []
    marker_idx = 0
    for method in methods_dict.values():

        if method in ['Autonomous AI', 'Aggressive AI', 'Conservative AI']:
            recall = PR_F1[method]['Recall']
            precision = PR_F1[method]['Precision']
            F1 = PR_F1[method]['F1']
            # marker = "d" # '^'

        else:
            condition = (df_difficulty['Method'] == method)
            df_method = df_difficulty.loc[condition]

            recall = np.mean(df_method['Recall'])
            precision = np.mean(df_method['Precision'])
            F1 = np.mean(df_method['F1'])
            # marker = 'o'

        F1_figure = 2*precision*recall/(precision+recall)
        print(f'{method}, Precision:{precision:.2f}, Recall:{recall:.2f}, F1(figure): {F1_figure:.2f}, F1(ture mean):{F1:.2f}')

        """
        Human Only, Precision:92.39, Recall:94.40, F1(figure): 93.38, F1(ture mean):93.25
        Autonomous AI, Precision:94.74, Recall:91.85, F1(figure): 93.27, F1(ture mean):93.04
        Aggressive AI, Precision:97.39, Recall:89.73, F1(figure): 93.40, F1(ture mean):93.24
        Aggressive AI + Human, Precision:96.34, Recall:97.43, F1(figure): 96.88, F1(ture mean):96.79
        Conservative AI, Precision:88.01, Recall:94.05, F1(figure): 90.93, F1(ture mean):90.28
        Conservative AI + Human, Precision:95.50, Recall:98.08, F1(figure): 96.78, F1(ture mean):96.66
        """

        pres.append(precision)
        recs.append(recall)

        plt.scatter(recall, precision, label=method) #, alpha=0.01)
        # put emoji markers
        # ab = AnnotationBbox(emojis[marker_idx], (recall, precision), frameon=False)
        # plt.gca().add_artist(ab)
        # marker_idx += 1
        # ax.set_title("Overall")

    plt.xticks(np.arange(85, 101, 5))
    plt.yticks(np.arange(85, 101, 5))

    plt.xlabel('Recall (%)')
    plt.ylabel('Precision (%)')

    # plot the arrows that connect 'Byte_json' to 'B' and 'Appen_json' to 'C'
    # if not part2:
    #     x0, y0, x1, y1 = recs[2]+0.5, pres[2], recs[3], pres[3]
    #     plt.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color='black', ls='--'))
    #     # axs[i].arrow(recall, precision, 0, 0.1, head_width=0.05, head_length=0.1, fc='k', ec='k')
    #     x0, y0, x1, y1 = recs[4], pres[4], recs[5], pres[5]
    #     plt.annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='-|>', color='black', ls='--'))

    plt.xlim(85, 100)
    plt.ylim(85, 100)

    # plt.legend(bbox_to_anchor=(0.0, 0.5), loc='upper center', borderaxespad=0, ncol=2)
    # handles, labels = ax.get_legend_handles_labels()
    plt.legend(loc='center left', bbox_to_anchor=(1.03, 0.5), borderaxespad=0, ncol=1)
    plt.tight_layout()

    plt.savefig(judge_dir+f'{prefix}_AB_testing_precision_recall_changes_overall.pdf', bbox_inches="tight")
    plt.savefig(judge_dir+f'{prefix}_AB_testing_precision_recall_changes_overall.png', bbox_inches="tight")


def concept_improvement_plot(judge_dir):
    sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
    plt.figure(figsize=(4.5, 4))

    # read emoji images as markers
    zm_s = 0.1
    zm_l = 0.1
    robot_green = OffsetImage(plt.imread('evaluation/robot_1f916_green.png'), zoom=zm_l)
    robot_orange = OffsetImage(plt.imread('evaluation/robot_1f916_orange.png'), zoom=zm_l)
    robot_grey = OffsetImage(plt.imread('evaluation/robot_1f916_grey.png'), zoom=zm_l)
    human_green = OffsetImage(plt.imread('evaluation/busts-in-silhouette_1f465_green.png'), zoom=zm_s)
    human_orange = OffsetImage(plt.imread('evaluation/busts-in-silhouette_1f465_orange.png'), zoom=zm_s)
    human_blue = OffsetImage(plt.imread('evaluation/busts-in-silhouette_1f465_blue.png'), zoom=zm_s)
    # emojis = [human_blue, human_orange, human_green, human_blue, robot_grey, robot_orange, robot_green]
    emojis = [robot_orange, robot_green]

    pres, recs = [80, 50], [50, 80]
    labels = ['Aggressive AI', 'Conservative AI']

    for method in range(2):
        recall = recs[method]
        precision = pres[method]
        plt.scatter(recall, precision, label=labels[method], alpha=0.01)
        # put emoji markers
        # ab = AnnotationBbox(emojis[method], (recall, precision), frameon=False)
        # plt.gca().add_artist(ab)

    # plt.xticks(np.arange(0, 101, 20))
    # plt.yticks(np.arange(0, 101, 20))

    plt.xlabel('Recall (%)')
    plt.ylabel('Precision (%)')

    plt.xlim(0, 100)
    plt.ylim(0, 100)

    # plt.legend(bbox_to_anchor=(0.0, 0.5), loc='upper center', borderaxespad=0, ncol=2)
    # handles, labels = ax.get_legend_handles_labels()
    # plt.legend(loc='center right', bbox_to_anchor=(-.2, 0.5), borderaxespad=0, ncol=1)
    plt.tight_layout()

    plt.savefig(judge_dir+f'concept_figure.pdf', bbox_inches="tight")


def improvement_plot(df_pr, PR_F1, judge_dir, difficulties, part2=False):
    # plot the transition from AI pre-annotation to rater judgements
    sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
    fig, axs = plt.subplots(1, len(difficulties), figsize=(12, 4), sharey=True, sharex=True)
    # save precision recall as x, y coordinates for arrow plotting
    precisions, recalls = [], []
    prefix = 'Part2' if part2 else 'Part1'

    vis_order = list(methods_dict.values())[:3]

    print()

    # plot the points
    for i in range(len(difficulties)):
        difficulty = difficulties[i]
        condition = (df_pr['Difficulty'] == difficulty)
        df_difficulty = df_pr.loc[condition]

        pres, recs = [], []
        for method in methods_dict.values():

            if method in ['FaceDetector', 'FaceDetector + HTM Tracker', 'FaceDetector + ByteTracker']:
                recall = np.mean(df_difficulty['Recall_'+method])
                precision = np.mean(df_difficulty['Precision_'+method])

                # vs. average over unique values
                # np.mean(df_method['Recall_'+method].unique()) 

            else:
                condition = (df_difficulty['Method'] == method)
                df_method = df_difficulty.loc[condition]

                recall = np.mean(df_method['Recall'])
                precision = np.mean(df_method['Precision'])

            print(f'{method} {difficulty}, recall:{recall:.2f}, precision:{precision:.2f}')
            pres.append(precision)
            recs.append(recall)
            axs[i].scatter(recall, precision, label=method)
            axs[i].set_title(difficulty)

        plt.xticks(np.arange(75, 101, 5))
        plt.yticks(np.arange(75, 101, 5))

        axs[i].set_xlabel('Recall (%)')

        precisions.append(pres)
        recalls.append(recs)
        print()

    # axs[0].set_xlabel('Recall (%)')
    axs[0].set_ylabel('Precision (%)')

    # plot the arrows that connect 'Byte_json' to 'B' and 'Appen_json' to 'C'
    for i in range(len(difficulties)):
        if part2: continue

        x0, y0, x1, y1 = recalls[i][3], precisions[i][3], recalls[i][1], precisions[i][1]
        axs[i].annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', color='black'))
        # axs[i].arrow(recall, precision, 0, 0.1, head_width=0.05, head_length=0.1, fc='k', ec='k')
        x0, y0, x1, y1 = recalls[i][4], precisions[i][4], recalls[i][2], precisions[i][2]
        axs[i].annotate('', xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle='->', color='black'))

        axs[i].set_xlim(75, 100)
        axs[i].set_ylim(75, 100)

    # plt.legend(bbox_to_anchor=(0.0, 0.5), loc='upper center', borderaxespad=0, ncol=2)
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left' , bbox_to_anchor=(0.5, 1.0), borderaxespad=0, ncol=2)
    plt.tight_layout()

    plt.savefig(judge_dir+f'{prefix}_AB_testing_precision_recall_changes.pdf')
    plt.savefig(judge_dir+f'{prefix}_AB_testing_precision_recall_changes.png')


def per_worker_plot(df_pr, judge_dir, order, cost_thresholds, part2=False):
    # plot the transition from AI pre-annotation to rater judgements
    sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
    prefix = 'Part2' if part2 else 'Part1'

    vis_order = list(methods_dict.values())[:3]

    # plot the points
    fig, axs = plt.subplots(1, len(order), figsize=(15, 3), sharey=True, sharex=True)
    for i in range(len(order)):
        difficulty = order[i]

        condition = (df_pr['Difficulty'] == difficulty)
        df_difficulty = df_pr.loc[condition]

        g = sns.stripplot(ax=axs[i], x="Recall", y="Method", data=df_difficulty, order=vis_order, dodge=True, alpha=.25, zorder=1)

        sns.pointplot(ax=axs[i], x="Recall", y="Method",
                    data=df_difficulty, order=vis_order, dodge=.8 - .8 / 3,
                    join=False, palette="dark",
                    markers="d", scale=.75, ci=None)

        g.set_xlim(75, 100)

        axs[i].set_title(difficulty)
        axs[i].set_xlabel('Recall (%)')
        axs[i].set_ylabel('')

    # plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0)
    plt.tight_layout()

    plt.savefig(judge_dir+f'{prefix}_AB_testing_recall_per_judgement.pdf')
    plt.savefig(judge_dir+f'{prefix}_AB_testing_recall_per_judgement.png')

def novice_veteran_plot(df_pr, judge_dir, order, cost_thresholds, part2=False):
    # plot the transition from AI pre-annotation to rater judgements
    sns.set(rc={"figure.dpi":300, 'savefig.dpi':300})
    prefix = 'Part2' if part2 else 'Part1'

    vis_order = ['Overall', 'Novice', 'Veteran']
    group_order = list(methods_dict.values())[:3]

    # plot the points
    fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True, sharex=True)

    condition_0 = (df_pr['Difficulty'] == 'Overall')

    for i in range(3):
        
        tenure = vis_order[i]

        if i==0:
            condition = condition_0
        else:
            condition = (df_pr['Difficulty'] == 'Overall') & (df_pr['tenure_group'] == tenure)

        df_temp = df_pr.loc[condition]

        g = sns.stripplot(ax=axs[i], x="Recall", y="Method", data=df_temp, order=group_order, dodge=True, alpha=.25, zorder=1)

        sns.pointplot(ax=axs[i], x="Recall", y="Method",
                    data=df_temp, order=group_order, dodge=.8 - .8 / 3,
                    join=False, palette="dark",
                    markers="d", scale=.75, ci=None)

        g.set_xlim(0, 100)

        axs[i].set_title(tenure)
        axs[i].set_xlabel('Recall (%)')
        axs[i].set_ylabel('')

    # plt.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', borderaxespad=0)
    plt.tight_layout()

    plt.savefig(judge_dir+f'{prefix}_AB_testing_recall_by_tenure.pdf')
    plt.savefig(judge_dir+f'{prefix}_AB_testing_recall_by_tenure.png')



if __name__ == "__main__":    

    plot_part2 = False
    consider_optional = True
    cost_thresholds = [0.5]
    # cost_thresholds = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    # cost_thresholds = np.linspace(0.1, 0.9, 9)

    base_dir = 'user_study_results/'

    if plot_part2:
        df_path = base_dir + 'part_2_progress_check/df2_combined.csv'
        judge_dir = base_dir + 'part_2_merged_judgements/'
        gt_dir = judge_dir + 'reviewed_judgements_optional_consensus_0.75_IOU_0.3/'

        # AI_methods = ['Retina_0.3']
        AI_methods = []
        methods_dict = {
            'A':'Human only, Group A',
            'B':'Human only, Group B',
            'C':'Human only, Group C',
            'Retina_0.3':'Autonomous AI',
            }
            # 'Retina_json':'FaceDetector'

    else:
        df_path = base_dir + 'part_1_progress_check/df_combined.csv'
        judge_dir = base_dir + 'part_1_merged_judgements/'
        gt_dir = judge_dir + 'reviewed_judgements_optional_consensus_0.75_IOU_0.3/'

        AI_methods = ['Appen_json', 'Byte_json', 'Retina_0.3']
        methods_dict ={
            'A':'Human Only, Group A',
            'B':'Restrained AI + Group B',
            'C':'Zealous AI + Group C',
            'Retina_0.3':'Autonomous AI',
            'Byte_json':'Restrained AI',
            'Appen_json':'Zealous AI',
            }
            # 'Retina_0.01':'Conservative FaceDetector',
            # 'Retina_0.8':'Aggressive FaceDetector',

    # gt_dir = judge_dir + 'part_1_judgements_merged_AC_improved/'
    # gt_dir = judge_dir + 'reviewed_judgements_v2/'
    # gt_dir = judge_dir + 'reviewed_judgements_with_optional/'

    df_save_path = judge_dir + f'AB_testing_precision_recall_IOU_{1-cost_thresholds[0]}.csv'

    if os.path.exists(df_save_path):
        df_pr = pd.read_csv(df_save_path)
    else:
        # read combined dataframe first then add new columns
        df = pd.read_csv(df_path)
        df_pr = df.copy()

        # do not exclude outliers from time analysis for Precision-Recall
        # df = df[df['Outlier']==0]

        clips = df_pr['Clip_name'].unique()
        gt_paths = glob.glob(gt_dir + '*.json')

        print(f'{len(clips)} clips considered for precision recall')

        # iterate over each judgement in df_pr
        for i, row in df_pr.iterrows():
            clip = row['Clip_name']
            worker_id = str(row['_worker_id'])
            method = row['Group']
            rater_json = row['Rater_json']

            # find the matching judgement json file
            if not os.path.isfile(rater_json):
                raise Exception(f'{rater_json} does not exist')

            # the matching GT json file's path
            # gt_json = gt_dir + f'{clip}_reviewed.json'
            # gt_json = gt_dir + f'{clip}_merged.json'
            gt_json = gt_dir + f'{clip}_with_optional.json'
            df_pr.loc[i, 'GT_json'] = gt_json

            for t in cost_thresholds:
                pre, rec = one_video_precision_recall(gt_json, [rater_json], [t], consider_optional)

                # calculate the F1 score for each video judgement
                f1 = 2 * pre * rec / (pre + rec)

                df_pr.loc[i, 'Precision'] = round(pre.item(), 5)
                df_pr.loc[i, 'Recall'] = round(rec.item(), 5)
                df_pr.loc[i, 'F1'] = round(f1.item(), 5)
                df_pr.loc[i, 'Method'] = methods_dict[method]
                df_pr.loc[i, 'IOU_threshold'] = 1-t
    
            # collect precision and recall for APPEN and BYTE pre-annotation
            for AI_method in AI_methods:
                AI_json = row[AI_method]

                for t in cost_thresholds:
                    pre, rec = one_video_precision_recall(gt_json, [AI_json], [t], consider_optional)

                    f1 = 2 * pre * rec / (pre + rec)
                    
                    df_pr.loc[i, 'Precision_' + methods_dict[AI_method]] = round(pre.item(), 5)
                    df_pr.loc[i, 'Recall_' + methods_dict[AI_method]] = round(rec.item(), 5)
                    df_pr.loc[i, 'F1_' + methods_dict[AI_method]] = round(f1.item(), 5)

        # box-cox transform on Duration, F1, Recall for later processing 
        for var in ["Duration", "F1", "Recall"]:
            pt = PowerTransformer(method='box-cox', standardize=True)
            data = df_pr[var].values.reshape(-1, 1)
            new_var = var + '_transformed'
            df_pr[new_var] = pt.fit_transform(data)
            print(f'old {var}: {df_pr[var].mean()} {df_pr[var].std()}')
            print(f'new {new_var}: {df_pr[new_var].mean()} {df_pr[new_var].std()}')

        # save updated df
        df_pr.to_csv(df_save_path, index=False)

    difficulties = ["Overall", "Easy", "Medium", "Hard"]

    # print the accurate Precision, Recall and F1 score for each AI pre-annotation
    # each method should be average over 24 videos only
    precision_recall_F1_AI = {}

    for m in AI_methods:
        method = methods_dict[m]
        precision_recall_F1_AI[method] = {}
        precisions, recalls, f1s = [], [], []
        
        for clip in df_pr.Clip_name.unique():
            df_clip = df_pr[df_pr['Clip_name']==clip]
            precisions.append(df_clip['Precision_' + method].values[0])
            recalls.append(df_clip['Recall_' + method].values[0])
            f1s.append(df_clip['F1_' + method].values[0])
        precision_recall_F1_AI[method]['Precision'] = np.mean(precisions)
        precision_recall_F1_AI[method]['Recall'] = np.mean(recalls)
        precision_recall_F1_AI[method]['F1'] = np.mean(f1s)

    print()

    # add overall to difficulty
    df_copy = df_pr.copy()
    df_copy['Difficulty'] = 'Overall'
    df_overall = pd.concat([df_pr, df_copy])

    # if filter out low recall outliers
    # df_overall = df_overall[df_overall['Recall']>50]

    # concept_improvement_plot(judge_dir)

    # improvement_plot(df_overall, precision_recall_F1_AI, judge_dir, difficulties[:1], part2=plot_part2)

    # improvement_plot_single(df_overall, precision_recall_F1_AI, judge_dir, part2=plot_part2)

    per_worker_plot(df_overall, judge_dir, difficulties, cost_thresholds, part2=plot_part2)

    novice_veteran_plot(df_overall, judge_dir, difficulties, cost_thresholds, part2=plot_part2)

    # per_judgement_plot_each_method(df_pr, judge_dir, part2=plot_part2)

    # comparing just AI initializations
    # df_overall_AI = df_overall[(df_overall['Method'] == 'Aggressive AI') & (df_overall['Method'] == 'Conservative AI')]
    
    breakpoint()

    ############# Revision ananylisis #############

    pg.homoscedasticity(data=df_pr, dv='F1', group='Group')

    # df_pr['F1_box'] = scipy.stats.boxcox(f1, -3)
    # pg.homoscedasticity(data=df_pr, dv='F1_box', group='Group')
    #                W     pval  equal_var
    # levene  2.272688  0.10332       True



    ############## Initial submission ##############

    # compare each team with their AI initialization
    df_AI_only = pd.melt(df_pr, id_vars=['Group'], value_vars=['F1_Aggressive AI', 'F1_Conservative AI', 'F1'])
    df_B_only = df_AI_only[df_AI_only['Group'] == 'B']
    df_B_only = df_B_only.loc[~df_B_only['variable'].str.contains('Conservative')]

    pg.homoscedasticity(data=df_B_only, dv='value', group='variable')
    'false'
    pg.welch_anova(data=df_B_only, dv='value', between='variable')
    """
         Source  ddof1        ddof2          F         p-unc       np2
    0  variable      1  1228.173106  178.39228  4.298412e-38  0.125594
    """

    df_AI_only = pd.melt(df_pr, id_vars=['Group'], value_vars=['F1_Aggressive AI', 'F1_Conservative AI', 'F1'])
    df_C_only = df_AI_only[df_AI_only['Group'] == 'C']
    df_C_only = df_C_only.loc[~df_C_only['variable'].str.contains('Aggressive')]
    pg.homoscedasticity(data=df_C_only, dv='value', group='variable')
    'false'
    pg.welch_anova(data=df_C_only, dv='value', between='variable')
    """
         Source  ddof1       ddof2           F         p-unc       np2
    0  variable      1  837.860341  169.744837  1.820921e-35  0.120408
    """


    # only consider the F1 score of Aggressive AI and Conservative AI pre-annotations
    df_AI_only = pd.melt(df_pr, id_vars=['Group'], value_vars=['F1_Aggressive AI', 'F1_Conservative AI', 'F1'])
    df_AI_B_only = df_AI_only.loc[df_AI_only['variable'].str.contains('Aggressive') & df_AI_only['Group'].str.contains('B')]
    df_AI_C_only = df_AI_only.loc[df_AI_only['variable'].str.contains('Conservative') & df_AI_only['Group'].str.contains('C')]
    df_BC_f1 = df_AI_B_only.append(df_AI_C_only)

    pg.homoscedasticity(data=df_BC_f1, dv='value', group='Group')
    """
                    W          pval  equal_var
    levene  69.146354  2.377549e-16      False
    """
    pg.welch_anova(data=df_BC_f1, dv='value', between='Group')
    """
      Source  ddof1       ddof2          F         p-unc       np2
    0  Group      1  854.045981  35.320502  4.071011e-09  0.027703
    """

    # calculate the F1 improvements for B and C
    for idx, row in df_pr.iterrows():
        if 'B' in row['Group']:
            df_pr.loc[idx, 'F1_improvement'] = row['F1'] - row['F1_Aggressive AI']
        elif 'C' in row['Group']:
            df_pr.loc[idx, 'F1_improvement'] = row['F1'] - row['F1_Conservative AI']
    df_BC_f1_improve = pd.melt(df_pr, id_vars=['Group'], value_vars=['F1_improvement'])
    df_BC_f1_improve = df_BC_f1_improve[df_BC_f1_improve['Group']!='A']
    breakpoint()
    pg.homoscedasticity(data=df_BC_f1_improve, dv='value', group='Group')
    """
                    W          pval  equal_var
    levene  55.418322  1.815646e-13      False
    """
    pg.welch_anova(data=df_BC_f1_improve, dv='value', between='Group')
    """
      Source  ddof1      ddof2          F         p-unc      np2
    0  Group      1  934.06568  45.022824  3.372223e-11  0.03504
    """


    # F1 between groups
    pg.homoscedasticity(data=df_pr, dv='F1', group='Group')
    """
    Part 1 Overall
                    W          pval  equal_var
    levene  36.998343  1.748739e-16      False
    """
    pg.welch_anova(data=df_pr, dv='F1', between='Group')
    """
      Source  ddof1        ddof2          F         p-unc      np2
    0  Group      2  1147.898036  23.151595  1.389584e-10  0.03963
    """
    # jsut compare the two human-AI groups
    pg.welch_anova(data=df_pr[df_pr['Group']!='A'], dv='F1', between='Group')
    """
       Source  ddof1        ddof2         F     p-unc       np2
    0  Method      1  1235.145746  0.244073  0.621367  0.000197
    """
    pg.pairwise_gameshowell(data=df_pr, dv='F1', between='Group')
    """
       A  B    mean(A)    mean(B)      diff        se         T           df          pval    hedges
    0  A  B  93.251837  96.788625 -3.536788  0.526400 -6.718818   780.697439  1.061675e-10 -0.380606
    1  A  C  93.251837  96.659524 -3.407686  0.530795 -6.419964   802.898845  6.991668e-10 -0.363823
    2  B  C  96.788625  96.659524  0.129102  0.261320  0.494038  1235.145746  8.741696e-01  0.028009
    """


    # test for equality of variances
    out = pg.homoscedasticity(data=df, dv='Recall', group='Group')
    print(out)
    """
    Part 1 Overall
                    W          pval  equal_var
    levene  38.590797  3.785273e-17      False
    Easy
    levene  5.588375  0.003998      False
    Medium
    levene  15.368249  2.941912e-07      False

    Part 2 Overall
                   W      pval  equal_var
    levene  9.909964  0.000055      False
    Easy
    levene  3.01481  0.051041       True
    Medium
    levene  3.791951  0.023636      False
    Hard
    levene  3.733435  0.024794      False
    """

    aov_welch = pg.welch_anova(data=df, dv='Recall', between='Group')
    print(aov_welch)
    """
    Part 1 Overall
      Source  ddof1        ddof2          F         p-unc       np2
    0  Group      2  1107.685618  27.670074  1.877527e-12  0.040554

    Part 2 Overall
      Source  ddof1       ddof2         F     p-unc       np2
    0  Group      2  507.011266  9.649509  0.000077  0.020968
    Medium
    0  Group      2  181.160677  3.219826  0.04226  0.023916
    Hard
    0  Group      2  209.214146  4.028427  0.019201  0.018611
    """

    gamesshowell = pg.pairwise_gameshowell(data=df, dv='Recall', between='Group')
    print(gamesshowell)
    """
    Part 1 Overall
       A  B    mean(A)    mean(B)      diff        se         T           df          pval    hedges
    0  A  B  94.398404  97.429871 -3.031466  0.527344 -5.748559   755.192358  3.915215e-08 -0.325643
    1  A  C  94.398404  98.082546 -3.684142  0.516448 -7.133616   700.065048  7.231438e-12 -0.404266
    2  B  C  97.429871  98.082546 -0.652676  0.207345 -3.147771  1160.299614  4.800313e-03 -0.178458

    Part 1 Easy
       A  B    mean(A)    mean(B)      diff        se         T          df      pval    hedges
    0  A  B  95.872310  97.676158 -1.803847  0.815191 -2.212790  187.943069  0.071614 -0.250743
    1  A  C  95.872310  97.995106 -2.122796  0.809677 -2.621781  183.485231  0.025563 -0.296614
    2  B  C  97.676158  97.995106 -0.318948  0.352772 -0.904121  307.260662  0.638110 -0.102287

    Part 1 Medium
       A  B    mean(A)    mean(B)      diff        se         T          df      pval    hedges
    0  A  B  93.530707  96.794200 -3.263493  0.816322 -3.997803  289.344242  0.000239 -0.369002
    1  A  C  93.530707  97.292759 -3.762053  0.813085 -4.626885  285.342360  0.000017 -0.427983
    2  B  C  96.794200  97.292759 -0.498559  0.374724 -1.330471  463.487774  0.378957 -0.123068

    Part 1 Hard
       A  B    mean(A)    mean(B)      diff        se         T          df      pval    hedges
    0  A  B  94.289797  97.904431 -3.614634  1.003483 -3.602088  277.429838  0.001086 -0.332832
    1  A  C  94.289797  98.927488 -4.637691  0.965699 -4.802421  240.104455  0.000008 -0.443743
    2  B  C  97.904431  98.927488 -1.023057  0.320031 -3.196744  303.715041  0.004371 -0.295694

    Part 2 Overall
       A  B    mean(A)    mean(B)      diff        se         T          df      pval    hedges
    0  A  B  94.143196  96.904655 -2.761459  1.082215 -2.551673  409.130199  0.029759 -0.205548
    1  A  C  94.143196  98.069188 -3.925992  1.025747 -3.827445  338.596119  0.000452 -0.309615
    2  B  C  96.904655  98.069188 -1.164532  0.460532 -2.528666  460.277254  0.031558 -0.205700

    Part 2 Medium
       A  B    mean(A)    mean(B)      diff        se         T          df      pval    hedges
    0  A  B  94.662863  97.625007 -2.962144  1.716555 -1.725633  113.541628  0.200107 -0.240785
    1  A  C  94.662863  98.391244 -3.728382  1.704041 -2.187965  110.419807  0.077701 -0.305296
    2  B  C  97.625007  98.391244 -0.766238  0.495092 -1.547666  192.140615  0.271114 -0.218043

    Part 2 Hard
       A  B    mean(A)    mean(B)      diff        se         T          df      pval    hedges
    0  A  B  92.825644  95.512169 -2.686524  1.914457 -1.403282  192.130507  0.341217 -0.174222
    1  A  C  92.825644  97.139431 -4.313787  1.755399 -2.457439  144.728680  0.040005 -0.307554
    2  B  C  95.512169  97.139431 -1.627263  0.969360 -1.678697  182.967067  0.216127 -0.210887

    """

    print(pg.anova(data=df, dv='Recall', between='Group', detailed=True))

    """
    Part 2 Easy
       Source            SS   DF          MS        F     p-unc     np2
    0   Group    509.798460    2  254.899230  3.01493  0.051035  0.0261
    1  Within  19022.772826  225   84.545657      NaN       NaN     NaN
    """

    print(pg.pairwise_tukey(data=df, dv='Recall', between='Group'))

    """
    Part 2 Easy
       A  B    mean(A)    mean(B)      diff        se         T   p-tukey    hedges
    0  A  B  95.665746  98.302065 -2.636319  1.486754 -1.773204  0.180961 -0.285290
    1  A  C  95.665746  99.176976 -3.511230  1.491734 -2.353790  0.050691 -0.379956
    2  B  C  98.302065  99.176976 -0.874911  1.496569 -0.584611  0.828504 -0.094672
    """
