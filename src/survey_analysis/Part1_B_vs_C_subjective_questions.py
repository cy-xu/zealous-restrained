from collections import OrderedDict
import copy
import pingouin as pg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec


category_names = ['Strongly Prefer AI', 'Prefer AI',
                  'No preference', 'Prefer Manual', 'Strongly Prefer Manual']

questions = {
    'S1-3: To what extent do you trust the pre-annotation from the AI?': {'B':[0,0,0,0,0], 'C':[0,0,0,0,0],},
    'S1-4: How consistent do you think the AI is?': {'B':[0,0,0,0,0], 'C':[0,0,0,0,0],},
    'S1-5: Does the pre-annotation from the AI often surprise you?': {'B':[0,0,0,0,0], 'C':[0,0,0,0,0],},
    }

questions_inverse = {
    'S1-1: Do you think the AI is correct most of the time?': {'B':[0,0,0,0,0], 'C':[0,0,0,0,0],},
    'S1-2: Do you think the AI often makes mistakes?': {'B':[0,0,0,0,0], 'C':[0,0,0,0,0],},
    'S1-6: Do you think working with this AI makes your job easier than you annotate manually?': {'B':[0,0,0,0,0], 'C':[0,0,0,0,0],},
    'S1-7: Given two different AI models, which one do you prefer to work with?':{'B': [0,0,0,0,0], 'C':[0,0,0,0,0]},
}

"""
answer key for the last question
A. It can find most of the faces but you need to fix many mistakes it makes
B. It doesn’t make many mistakes but you need to add many missing faces
"""

task_difficulty = {
    "Do you think the task of face annotation is easy or hard?": {
        "A": [0, 0, 0, 0, 0], "B": [0, 0, 0, 0, 0], "C": [0, 0, 0, 0, 0]
        },}

convert_dict = {'(1)': 0, '(2)': 1, '(3)': 2, '(4)': 3, '(5)': 4}
convert_dict_inverse = {'(1)': 4, '(2)': 3, '(3)': 2, '(4)': 1, '(5)': 0}


def get_answer_count(df, questions, convert_dict, exclude_Q_idx):
    questions_100 = copy.deepcopy(questions)

    for idx, row in df.iterrows():
        for q in questions.keys():
            qst = q[6:] if exclude_Q_idx else q
            for choice in convert_dict.keys():
                if choice in row[qst]:
                    questions[q][row['Group']][convert_dict[choice]] += 1

    # normalize vote counts to percentages
    for q in questions.keys():
        for g in questions[q].keys():
            total = np.sum(questions[q][g])
            questions_100[q][g] = [round(x/total*100, 2) for x in questions[q][g]]

    return questions, questions_100

def survey(result_counts, results, category_names, rows, cols, bar_h, vertical, fig_size=(10, 14)):
    """
    Parameters
    ----------
    results : dict
        A mapping from question labels to a list of answers per category.
        It is assumed all lists contain the same number of entries and that
        it matches the length of *category_names*. The order is assumed
        to be from 'Strongly disagree' to 'Strongly aisagree'
    category_names : list of str
        The category labels.
    """
    
    # Color Mapping
    category_colors = plt.get_cmap('coolwarm_r')(np.linspace(0.15, 0.85, 5))
    
    # fig, axs = plt.subplots(len(questions), 1, figsize=(6, 10))

    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(nrows=rows, ncols=cols, hspace=0.8)

    for ax_i in range(len(results)):
        ax = fig.add_subplot(gs[ax_i])
        question = list(results.keys())[ax_i]

        labels = list(results[question].keys())
        data = np.array(list(results[question].values()))
        data_cum = data.cumsum(axis=1)
        middle_index = data.shape[1]//2
        offsets = data[:, range(middle_index)].sum(axis=1) + data[:, middle_index]/2

        ### plot mean and 95% CI
        answers = np.array(list(result_counts[question].values()))
        # weight = [1, 2, 3, 4, 5]
        weight = [-2, -1, 0, 1, 2]
        # weight = [3, 2, 1, 2, 3]
        
        # repeat each weight N times for N in answers
        repeated_weights = []
        for i in range(len(answers)):
            repeated = np.repeat(weight, answers[i])
            repeated_weights.append(repeated)
        repeated_weights = np.array(repeated_weights)

        weighted_answers = answers * weight
        weighted_cum = weighted_answers.cumsum(axis=1)
        # convert to 100 percent scale
        weighted_percent = weighted_cum / np.sum(weighted_answers, axis=1)[:, None] * 100
        median_xs = [None, None]

        # Plot Bars
        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths - offsets
            ax.barh(vertical, widths, left=starts, height=bar_h,
                        label=colname, color=color, align='center', edgecolor='white')

            for row in range(len(data)):
                # weighted median
                if weighted_percent[row, i] >= 50. and median_xs[row] is None:
                    # then this section is where the 50% mark lies in
                    diff = widths[row] - (weighted_percent[row, i] - 50.)
                    median_xs[row] = starts[row] + diff

        # Add Zero Reference Line
        ax.axvline(0, linestyle='dotted', color='black', alpha=.25)

        # Add weighted median Reference Line based on Likert scale
        # for bar_i in range(len(data)):
            # weighted median (needs to be explained and probably unconventional)
            # ax.axvline(median_xs[bar_i], linestyle='dotted', color='red', alpha=1.0)

        # un-weighted median
        # y_min = np.array(vertical) - bar_h/2
        # y_max = np.array(vertical) + bar_h/2
        # ax.vlines(50-offsets, y_min, y_max, linestyle='-', color='tab:red', alpha=0.8)

        # plot the mean and 95% CI for each group


        # this is how un-weighted median is calculated
        # https://community.tableau.com/s/question/0D54T00000C5dwESAR/calculating-the-median-of-likert-scale-data-that-assumes-that-the-5-point-scale-represents-a-continuous-random-variable

        # X Axis
        ax.set_xlim(-75, 75)
        ax.set_xticks(np.arange(-75, 75+1, 25), alpha=0.25)
        # ax.xaxis.set_major_formatter(lambda x, pos: str(abs(int(x))))
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=100, symbol="%"))
        plt.setp(ax.get_xticklabels(), alpha=0.5)
        # set title for this subplot

        ax.set_yticks(vertical)
        ax.set_yticklabels(labels, fontsize=12)

        # set title under the figure
        # ax.set_title(question, y=-0.7, fontsize=12)
        # set the tile above the figure
        ax.set_title(question, y=1.0, fontsize=12)

        # Y Axis
        ax.invert_yaxis()
        
        # Remove spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set(alpha=0.5)
    
        # Ledgend
        # if ax_i == 0:
        #     ax.legend(ncol=5, bbox_to_anchor=(0.5, 1.3), loc='lower center')
            # , fontsize='small',

    # Set Background Color
    fig.set_facecolor('#FFFFFF')
    # fig.tight_layout()

    return fig


def likert_and_mean_sd(result_counts, results, category_names, rows, cols, bar_h, vertical, fig_size=(10, 14)):
    # Color Mapping
    # category_colors = plt.get_cmap('coolwarm_r')(np.linspace(0.15, 0.85, 5))
    
    category_colors = ['darkorange' ,'mediumseagreen']

    fig = plt.figure(figsize=fig_size)
    gs = gridspec.GridSpec(nrows=rows, ncols=cols, hspace=0.8)

    for ax_i in range(len(results)):
        ax = fig.add_subplot(gs[ax_i])
        question = list(results.keys())[ax_i]

        ### plot mean and 95% CI
        answers = np.array(list(result_counts[question].values()))
        weight = [-2, -1, 0, 1, 2]
        max_count = max(np.sum(answers, axis=1))

        # repeat each weight N times for N in answers
        repeated_weights = []
        for i in range(len(answers)):
            repeated = np.repeat(weight, answers[i]).tolist()
            if len(repeated) < max_count: repeated.extend([0]*(max_count-len(repeated)))
            repeated_weights.append(repeated)
        repeated_weights = np.array(repeated_weights)

        vertical = [0.6, 0.2]
 
        # plot horizontal lines for mean and 95% CI
        # repeated_weights has shape [2, 26]
        means = np.mean(repeated_weights, axis=1)
        ci = 1.96 * np.std(repeated_weights, axis=1) / np.sqrt(max_count)
        # plot the error bar and 95 CI for each group
        for i in range(len(means)):
            color = category_colors[i]
            ax.errorbar(means[i], vertical[i], xerr=ci[i], color=color, fmt='o', elinewidth=3, capsize=6, capthick=1, alpha=1.0)

        # Add Zero Reference Line
        ax.axvline(0, linestyle='dotted', color='black', alpha=.25)

        # X Axis
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(0, 1)
        ax.set_title(question, y=1.0, fontsize=12)

        # Remove spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set(alpha=0.5)
        ax.set_yticklabels([])
        ax.set_yticks([])

    # Set Background Color
    fig.set_facecolor('#FFFFFF')
    # fig.tight_layout()

    return fig

df = pd.read_csv('subjective_answers_summary.csv')
# filter part 1
df = df[(df['Part'] == 1)]

# 1 task difficulty question for ABC
bar_h = 0.1
vertical = [0.1, 0.2, 0.3]

# task_q = get_answer_count(df, task_difficulty, convert_dict, False)
# fig = survey(task_q, category_names, 1, 1, bar_h, vertical, fig_size=(10, 3))
# fig.savefig('fig_task_difficulty_question.pdf', bbox_inches='tight')

# filter Group B, C for these questions
df = df[(df['Group'] == 'B') | (df['Group'] == 'C')]
bar_h = 0.1
vertical = [0.1, 0.2]

# 7 questions between B & C
qst, qst_100 = get_answer_count(df, questions, convert_dict, True)
qst_inverse, qst_inverse_100 = get_answer_count(df, questions_inverse, convert_dict_inverse, True)

# combine questions and questions_inverse as one dict
qst.update(qst_inverse)
qst_100.update(qst_inverse_100)
# and sort the questiosn by keys
qst = OrderedDict(sorted(qst.items()))
qst_100 = OrderedDict(sorted(qst_100.items()))

breakpoint()

q_6 = 'S1-6: Do you think working with this AI makes your job easier than you annotate manually?'
qst_6 = OrderedDict({q_6: qst[q_6]})
qst_100_6 = OrderedDict({q_6: qst_100[q_6]})

fig = survey(qst, qst_100, category_names, 7, 1, bar_h, vertical, fig_size=(10, 14))
plt.savefig(f'fig_Part1_BC_subjective_questions.pdf', bbox_inches='tight')

fig = likert_and_mean_sd(qst_6, qst_100_6, category_names, 7, 1, bar_h, vertical, fig_size=(10, 14))
plt.savefig(f'fig_Part1_BC_survey_error_bars.pdf', bbox_inches='tight')


### statistical analysis

convert_dict = {'(1)': 0, '(2)': 1, '(3)': 2, '(4)': 3, '(5)': 4}
# check each row to replace answer with the convert_dict values
for idx, row in df.iterrows():
    for q in questions.keys():
        for choice in convert_dict.keys():
            if choice in row[q]:
                df.loc[idx, q] = convert_dict[choice]

# conver to avoid object arrays issue
df = pd.DataFrame(df.to_dict())

for qst in questions.keys():
    print(qst)

    # var = pg.homoscedasticity(data=df, dv=qst, group='Group')
    # print(var)

    B_values = df[df['Group']=='B'][qst].tolist()
    C_values = df[df['Group']=='C'][qst].tolist()

    # for categorical data, it is more appropriate to use mann whitney u test
    print(pg.mwu(B_values, C_values, alternative='two-sided'))

    # aov = pg.anova(data=df, dv=qst, between='Group')
    # print(aov)

    """
    Mann Whitney U Test
    ------------------
    Do you think the AI is correct most of the time?
        U-val alternative     p-val       RBC      CLES
    MWU  277.0   two-sided  0.323212  0.147692  0.426154

    Do you think the AI often makes mistakes?
        U-val alternative     p-val       RBC      CLES
    MWU  248.5   two-sided  0.127289  0.235385  0.382308

    To what extent do you trust the pre-annotation from the AI?
        U-val alternative     p-val       RBC      CLES
    MWU  377.5   two-sided  0.280508 -0.161538  0.580769

    How consistent do you think the AI is?
        U-val alternative     p-val       RBC      CLES
    MWU  363.5   two-sided  0.435854 -0.118462  0.559231

    Does the pre-annotation from the AI often surprise you?
        U-val alternative     p-val       RBC      CLES
    MWU  329.0   two-sided  0.944677 -0.012308  0.506154

    Do you think working with this AI makes your job easier than you annotate manually?
        U-val alternative     p-val       RBC      CLES
    MWU  276.0   two-sided  0.341947  0.150769  0.424615

    Given two different AI models, which one do you prefer to work with?
        U-val alternative     p-val       RBC      CLES
    MWU  349.0   two-sided  0.641386 -0.073846  0.536923


    ANOVA results:
    --------------
    Do you think the AI is correct most of the time?
    Source  ddof1  ddof2         F     p-unc       np2
    0  Group      1     49  0.642999  0.426499  0.012952

    Do you think the AI often makes mistakes?
    Source  ddof1  ddof2        F     p-unc      np2
    0  Group      1     49  2.51118  0.119475  0.04875

    To what extent do you trust the pre-annotation from the AI?
    Source  ddof1  ddof2         F   p-unc       np2
    0  Group      1     49  1.201908  0.2783  0.023941

    How consistent do you think the AI is?
    Source  ddof1  ddof2         F    p-unc      np2
    0  Group      1     49  0.766902  0.38545  0.01541

    Does the pre-annotation from the AI often surprise you?
    Source  ddof1  ddof2         F     p-unc       np2
    0  Group      1     49  0.078075  0.781099  0.001591

    Do you think working with this AI makes your job easier than you annotate manually?
    Source  ddof1  ddof2       F     p-unc       np2
    0  Group      1     49  1.0334  0.314355  0.020654

    Given two different AI models, which one do you prefer to work with?
    Source  ddof1  ddof2         F     p-unc       np2
    0  Group      1     49  0.249346  0.619771  0.005063
    """