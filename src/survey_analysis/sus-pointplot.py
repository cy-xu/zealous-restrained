import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg

# Load data from csv file into NumPy dataframe
df = pd.read_csv('SUS_scores_total.csv')
# df = df[df['Part']=='Part 1']

plt.figure()
sns.set(rc={'figure.figsize':(3,2)})

plot_condition = (df['Part'] == 'Part 1')
plot_data = df[plot_condition]

ax = sns.pointplot(data=plot_data, x='Group', y='SUS score', dodge=True, capsize=.1, ci=95, seed=0)
ax.set(xlabel='Group', ylabel='SUS score') #, title='SUS scores comparison')
plt.ylim(0.0, 75.0)
# plt.legend(loc='lower right')
plt.savefig(f'fig_SUS_pointplot.pdf')
plt.savefig(f'fig_SUS_pointplot.png')
plt.clf() # this clears the figure

breakpoint()

for g in ['A', 'B', 'C']:
    condition = (df['Group'] == g) & (df['Part'] == 'Part 1')
    print(f'Part 1, {g} SUS score: {df.loc[condition, "SUS score"].mean()}')

for part in ['Part 1', 'Part 2']:
    condition = (df['Part'] == part)
    df_part = df.loc[condition, :] 

    # test for equality of variances
    out = pg.homoscedasticity(data=df_part, dv='SUS score', group='Group')
    print(out)
    """
    Part 1
                   W      pval  equal_var
    levene  0.523688  0.594482       True

    Part 2
                   W      pval  equal_var
    levene  0.309075  0.735123       True
    """

    aov = pg.anova(data=df_part, dv='SUS score', between='Group')
    """
    Part 1
      Source  ddof1  ddof2         F     p-unc      np2
    0  Group      2     75  1.240866  0.295001  0.03203

    Part 2
      Source  ddof1  ddof2        F     p-unc       np2
    0  Group      2     70  0.31649  0.729739  0.008962
    """



