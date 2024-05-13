import pandas as pd
import numpy as np


base_dir = "/Users/cyxu/Documents/github/zealous-restrained/user_study_results/"
judge_dir = base_dir + 'part_1_merged_judgements/'
cost_thresholds = [0.5]
prefix = 'Part1'

df_save_path = judge_dir + f'AB_testing_precision_recall_IOU_{1-cost_thresholds[0]}.csv'
df_save_clean = judge_dir + f'{prefix}_data_clean.csv'

df_pr = pd.read_csv(df_save_path)
df_clean = pd.read_csv(df_save_clean)

methods_dict ={
'A':'Human Only, Group A',
'B':'Restrained AI + Group B',
'C':'Zealous AI + Group C',
'Retina_0.3':'Autonomous AI',
'Byte_json':'Restrained AI',
'Appen_json':'Zealous AI',
}

for method in methods_dict.values():

    if method in ['Autonomous AI', 'Restrained AI', 'Zealous AI']:
        # recall = PR_F1[method]['Recall']
        # precision = PR_F1[method]['Precision']
        # F1 = PR_F1[method]['F1']
        recall = 1
        precision = 1
    else:
        condition = (df_clean['Method'] == method)
        df_method = df_clean.loc[condition]

        # unique entries count
        print(f'{method} entries: {len(df_method)}')

        recall_mean = np.mean(df_method['Recall']).round(3)
        recall_std = np.std(df_method['Recall']).round(3)

        precision_mean = np.mean(df_method['Precision']).round(3)
        precision_std = np.std(df_method['Precision']).round(3)

        f1_mean = np.mean(df_method['F1']).round(3)
        f1_std = np.std(df_method['F1']).round(3)

    F1_figure = 2*precision_mean*recall_mean/(precision_mean+recall_mean)
    F1_figure = round(F1_figure, 3)

    print(f'{method} recall_mean: {recall_mean}, recall_std: {recall_std}, precision_mean: {precision_mean}, precision_std: {precision_std}, f1_mean: {f1_mean}, f1_std: {f1_std}, F1_figure: {F1_figure} \n')

    # good, confirm F1 is same as paper
    # Human Only, Group A F1_figure: 94.52857053813062
    # Restrained AI + Group B F1_figure: 96.8795039264094
    # Zealous AI + Group C F1_figure: 96.77644673264186

