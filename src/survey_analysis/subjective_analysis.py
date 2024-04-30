import pandas as pd
import pingouin as pg

import seaborn as sns
import matplotlib.pyplot as plt


convert_dict = {'(1)': 0.0, '(2)': 1.0, '(3)': 2.0, '(4)': 3.0, '(5)': 4.0}

df = pd.read_csv('subjective_answers_summary.csv')

questions = [
    'Which method do you think is faster for face annotation?',
    'If we need the most accurate results for future videos, which method do you think is more reliable?',
    'Which method do you feel more relaxed to work with?',
    'Overall, which method do you prefer to use for future similar annotation jobs?'
    ]

def convert_to_score(string):
    for choice in convert_dict.keys():
        if choice in string:
            return convert_dict[choice]
    raise ValueError('No valid choice found in string')

for idx, row in df.iterrows():
    for qst in questions:
        df.at[idx, qst] = convert_to_score(row[qst])

# conver to avoid object arrays issue
df = pd.DataFrame(df.to_dict())

# melt to long format
df_long = df.melt(id_vars=['Group', 'Part'], value_vars=questions)

# # Set up the matplotlib figure
# f, ax = plt.subplots(figsize=(6, 4))
# # Draw a violinplot with a narrower bandwidth than the default
# sns.violinplot(data=df, x=questions, hue='Group', split=True, inner="stick", bw=.2)
# # Finalize the figure
# # ax.set(ylim=(-.7, 1.05))
# sns.despine(left=True, bottom=True)
# # ax.set(xlabel='Group', ylabel='SUS score') #, title='SUS scores comparison')
# # plt.ylim(0.0, 75.0)
# plt.legend(loc='lower right')
# plt.savefig(f'fig_SUS_pointplot.pdf')
# # plt.savefig(f'fig_SUS_pointplot.png')
# plt.clf() # this clears the figure


# we should not compare group B and C on these questions because
# they both prefer 

def run_anova(df, qst):
    # check for equal variance
    var = pg.homoscedasticity(data=df, dv=qst, group='Group')
    print(var)

    aov = pg.anova(data=df, dv=qst, between='Group')
    print(aov)

    # post-hoc
    # tukey = pg.pairwise_tukey(data=df, dv=qst, between='Group').round(3)
    # print(tukey)
    """
       A  B  mean(A)  mean(B)   diff     se      T  p-tukey  hedges
    which faster?
    0  B  C      2.4    2.286  0.114  0.456  0.251    0.803   0.077
    more reliable?
    0  B  C      2.4    2.667 -0.267  0.44  -0.606    0.548  -0.186
    more relaxed?
    0  B  C     2.55    2.571 -0.021  0.448 -0.048    0.962  -0.015
    prefer for future?
    0  B  C      2.4    2.619 -0.219  0.474 -0.462    0.647  -0.142
    """

def mwu_between_BC(df):
    for qst in questions:
        print(qst)
        group_b = df[df['Group'] == 'B'][qst]
        group_c = df[df['Group'] == 'C'][qst]
        print(pg.mwu(x=group_b, y=group_c, alternative='two-sided'))

        """
        which faster
            U-val alternative     p-val       RBC      CLES
        MWU  216.0   two-sided  0.883002 -0.028571  0.514286

        more reliable?
            U-val alternative     p-val  RBC  CLES
        MWU  189.0   two-sided  0.580668  0.1  0.45

        more relaxed?
            U-val alternative     p-val       RBC      CLES
        MWU  209.0   two-sided  0.989229  0.004762  0.497619

        prefer for future?
            U-val alternative     p-val       RBC      CLES
        MWU  198.5   two-sided  0.764964  0.054762  0.472619
        """
        print()

def plot_subjective_questions(df):
    # create new plt figure 
    fig = plt.figure()
    # Draw a nested boxplot to show bills by day and time
    # ax = sns.violinplot(x="value", y="variable", hue="Group", palette="Set2",
                    # data=df, split=True, scale="count")
    
    # sns.boxplot(x="value", y="variable", hue="Group", data=df)

    g = sns.catplot(x="Group", y="value", col="variable",
                data=df, kind="box",
                height=4, aspect=.7);

    # sns.despine(offset=10, trim=True)
    # save the figure
    plt.savefig(f'fig_subjecive_compare_BC.png')
    breakpoint()

plot_subjective_questions(df_long)
