import pingouin as pg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as gridspec


category_names = ['Strongly Prefer AI', 'Prefer AI',
                  'No preference', 'Prefer Manual', 'Strongly Prefer Manual']

questions = {
    'Which method is faster?': {'Group B':[0,0,0,0,0], 'Group C':[0,0,0,0,0],},
    'More reliable?': {'Group B':[0,0,0,0,0], 'Group C':[0,0,0,0,0],},
    'More relaxed to use?': {'Group B':[0,0,0,0,0], 'Group C':[0,0,0,0,0],},
    'Prefer to use in similar jobs?': {'Group B':[0,0,0,0,0], 'Group C':[0,0,0,0,0],},
    }

convert_dict = {'(1)': 0, '(2)': 1, '(3)': 2, '(4)': 3, '(5)': 4}
df = pd.read_csv('subjective_answers_P2_simple_questions.csv')

for idx, row in df.iterrows():
    for q in questions.keys():
        for choice in convert_dict.keys():
            if choice in row[q]:
                questions[q][row['Group']][convert_dict[choice]] += 1

# # normalize vote counts to percentages
# for q in questions.keys():
#     for g in ['Group B', 'Group C']:
#         total = np.sum(questions[q][g])
#         questions[q][g] = [round(x/total*100, 2) for x in questions[q][g]]

def survey(results, category_names):
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
    
    bar_h = 0.1
    vertical = [0.1, 0.2]

    # Color Mapping
    category_colors = plt.get_cmap('coolwarm_r')(
        np.linspace(0.15, 0.85, 5))
    
    # fig, axs = plt.subplots(len(questions), 1, figsize=(6, 10))

    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(nrows=4, ncols=1, hspace=0.8)

    for ax_i in range(len(questions)):
        ax = fig.add_subplot(gs[ax_i])
        question = list(results.keys())[ax_i]

        labels = list(results[question].keys())
        data = np.array(list(results[question].values()))
        data_cum = data.cumsum(axis=1)
        middle_index = data.shape[1]//2
        offsets = data[:, range(middle_index)].sum(axis=1) + data[:, middle_index]/2
 
        # Plot Bars
        for i, (colname, color) in enumerate(zip(category_names, category_colors)):
            widths = data[:, i]
            starts = data_cum[:, i] - widths - offsets
            ax.barh(vertical, widths, left=starts, height=bar_h,
                        label=colname, color=color, align='center', edgecolor='white')

        # Add Zero Reference Line
        ax.axvline(0, linestyle='dotted', color='black', alpha=.25)
        
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


def likert_and_mean_sd(result_counts):
    category_colors = ['darkorange' ,'mediumseagreen']

    fig = plt.figure(figsize=(6, 6))
    gs = gridspec.GridSpec(nrows=4, ncols=1, hspace=0.8)

    for ax_i in range(len(result_counts)):
        ax = fig.add_subplot(gs[ax_i])
        question = list(result_counts.keys())[ax_i]

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
        ax.set_xlim(-1.5, 1.5)
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


fig = survey(questions, category_names)
plt.savefig(f'fig_Part2_AI_vs_Manual_subjective.pdf', bbox_inches='tight')

# fig = likert_and_mean_sd(questions)
# plt.savefig(f'fig_Part2_survey_error_bars.pdf', bbox_inches='tight')

breakpoint()

### statistical analysis

# conver to avoid object arrays issue
df = pd.DataFrame(df.to_dict())

convert_dict = {'(1)': 0, '(2)': 1, '(3)': 2, '(4)': 3, '(5)': 4}
# check each row to replace answer with the convert_dict values
for idx, row in df.iterrows():
    for q in questions.keys():
        for choice in convert_dict.keys():
            if choice in row[q]:
                df.loc[idx, q] = convert_dict[choice]

for qst in questions.keys():
    print(qst)

    var = pg.homoscedasticity(data=df, dv=qst, group='Group')
    print(var)

    B_values = df[df['Group']=='Group B'][qst].tolist()
    C_values = df[df['Group']=='Group C'][qst].tolist()

    # for categorical data, it is more appropriate to use mann whitney u test
    print(pg.mwu(B_values, C_values, alternative='two-sided'))

    """
    Which method is faster?
        U-val alternative     p-val       RBC      CLES
    MWU  216.0   two-sided  0.883002 -0.028571  0.514286

    More reliable?
        U-val alternative     p-val  RBC  CLES
    MWU  189.0   two-sided  0.580668  0.1  0.45

    More relaxed to use?
        U-val alternative     p-val       RBC      CLES
    MWU  209.0   two-sided  0.989229  0.004762  0.497619

    Prefer to use in similar jobs?
        U-val alternative     p-val       RBC      CLES
    MWU  198.5   two-sided  0.764964  0.054762  0.472619
    """