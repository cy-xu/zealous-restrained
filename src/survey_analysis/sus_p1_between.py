import pandas as pd
import pingouin as pg

import matplotlib.pyplot as plt
import seaborn as sns


# Load data from csv file
df = pd.read_csv('sus-all-raw.csv')
questions = ['SUS1', 'SUS2', 'SUS3', 'SUS4', 'SUS5', 'SUS6', 'SUS7', 'SUS8', 'SUS9', 'SUS10']


def part1_anova(df):
    df_p1 = df[df['Part'] == 1]

    for q in questions:
        print(q)
        print(pg.homoscedasticity(df_p1, dv=q, group="Group"))

        print(pg.anova(df_p1, dv=q, between="Group"))
        print(pg.pairwise_tukey(df_p1, dv=q, between="Group").round(3))

        print(pg.welch_anova(df_p1, dv=q, between="Group"))
        print(pg.pairwise_gameshowell(df_p1, dv=q, between="Group").round(3))

        print()

        """
        Only Q4 is significant between A and B

        4. I think that I would need the support of a technical person to be able to use this system.
        Source  ddof1  ddof2         F     p-unc       np2
        0  Group      2     75  2.954545  0.058197  0.073034
        A  B  mean(A)  mean(B)   diff     se      T  p-tukey  hedges
        0  A  B    2.500    3.192 -0.692  0.287 -2.411    0.048  -0.659
        1  A  C    2.500    2.923 -0.423  0.287 -1.473    0.309  -0.403
        2  B  C    3.192    2.923  0.269  0.287  0.938    0.618   0.256
        """

def part12_within_subjects(df):
    B_1 = df[(df['Part'] == 1) & (df['Group'] == 'B')]
    B_2 = df[(df['Part'] == 2) & (df['Group'] == 'B')]
    C_1 = df[(df['Part'] == 1) & (df['Group'] == 'C')]
    C_2 = df[(df['Part'] == 2) & (df['Group'] == 'C')]

    for q in questions:
        print(q)
        print(pg.mwu(B_1[q], B_2[q], alternative='two-sided'))
        print(pg.mwu(C_1[q], C_2[q], alternative='two-sided'))


    """
    SUS1
         U-val alternative     p-val       RBC      CLES
    MWU  229.0   two-sided  0.467561  0.119231  0.440385
    MWU  222.5   two-sided  0.25852  0.184982  0.407509
    SUS2
    MWU  224.5   two-sided  0.405222  0.136538  0.431731
    MWU  257.5   two-sided  0.723236  0.056777  0.471612
    SUS3
    MWU  305.5   two-sided  0.296211 -0.175  0.5875
    MWU  260.5   two-sided  0.7882  0.045788  0.477106
    SUS4
    MWU  289.0   two-sided  0.513914 -0.111538  0.555769
    MWU  266.0   two-sided  0.884523  0.025641  0.487179
    SUS5
    MWU  255.5   two-sided  0.923868  0.017308  0.491346
    MWU  237.0   two-sided  0.418875  0.131868  0.434066
    SUS6
    MWU  269.5   two-sided  0.829484 -0.036538  0.518269
    MWU  284.5   two-sided  0.803613 -0.042125  0.521062
    SUS7
    MWU  309.5   two-sided  0.23109 -0.190385  0.595192
    MWU  225.5   two-sided  0.28616  0.173993  0.413004
    SUS8
    MWU  266.5   two-sided  0.88887 -0.025  0.5125
    MWU  210.5   two-sided  0.151213  0.228938  0.385531
    SUS9
    MWU  282.5   two-sided  0.608474 -0.086538  0.543269
    MWU  257.0   two-sided  0.724029  0.058608  0.470696
    SUS10
    MWU  338.5   two-sided  0.070853 -0.301923  0.650962
    MWU  251.0   two-sided  0.627336  0.080586  0.459707
    """


def combine_BC_to_compare_with_linear(df):
    df_BC1 = df[(df['Part'] == 1) & (df['Group'] != 'A')]
    df_BC2 = df[(df['Part'] == 2) & (df['Group'] != 'A')]

    # for q in questions:
    #     print(q)
    #     print(pg.mwu(df_BC1[q], df_BC2[q], alternative='two-sided'))

    """
    SUS1
        U-val alternative     p-val       RBC     CLES
    MWU  904.5   two-sided  0.185627  0.151501  0.42425
    SUS2
    MWU  963.0   two-sided  0.39001  0.096623  0.451689
    SUS3
    MWU  1131.5   two-sided  0.598444 -0.061445  0.530722
    SUS4
    MWU  1113.5   two-sided  0.706452 -0.044559  0.52228
    SUS5
    MWU  983.0   two-sided  0.494315  0.077861  0.461069
    SUS6
    MWU  1109.5   two-sided  0.722629 -0.040807  0.520403
    SUS7
    MWU  1066.0   two-sided    1.0  0.0   0.5
    SUS8
    MWU  958.5   two-sided  0.380592  0.100844  0.449578
    SUS9
    MWU  1075.0   two-sided  0.944729 -0.008443  0.504221
    SUS10
    MWU  1183.0   two-sided  0.347026 -0.109756  0.554878
    """

    for group in ['B', 'C']:
        df_g = df[(df['Group'] == group)]
        df_g_long = df_g.melt(id_vars=['Part'], value_vars=questions)

        # create new plt figure 
        fig = plt.figure()
        # Draw a nested boxplot to show bills by day and time
        ax = sns.violinplot(x="value", y="variable", hue="Part", palette="Set2",
                        data=df_g_long, split=True, scale="count")
        
        # sns.boxplot(x="value", y="variable", hue="Part", data=df_long)
        # sns.despine(offset=10, trim=True)
        # save the figure
        plt.savefig(f'fig_Group_{group}_SUS_within_subjects_violin.png')
        

def compare_BC_without_parts(df):
    df_B = df[(df['Group'] == 'B')]
    df_C = df[(df['Group'] == 'C')]

    for q in questions:
        print(q)
        print(pg.mwu(df_B[q], df_C[q], alternative='two-sided'))

    """
    Compare B and C but without differciate Part 1 and Part 2
        U-val alternative     p-val       RBC      CLES
    SUS1
    MWU  1121.5   two-sided  0.743999 -0.037465  0.518733
    SUS2
    MWU  1184.5   two-sided  0.391015 -0.095745  0.547872
    SUS3
    MWU  1063.0   two-sided  0.888028  0.016651  0.491674
    SUS4
    MWU  1153.5   two-sided  0.566684 -0.067068  0.533534
    SUS5
    MWU  1191.0   two-sided  0.367682 -0.101758  0.550879
    SUS6
    MWU  1083.5   two-sided  0.98692 -0.002313  0.501156
    SUS7
    MWU  1084.5   two-sided  0.980186 -0.003238  0.501619
    SUS8
    MWU  736.5   two-sided  0.005122  0.318686  0.340657
    SUS9
    MWU  995.5   two-sided  0.491168  0.079093  0.460453
    SUS10
    MWU  741.5   two-sided  0.006581  0.314061  0.342969
    """

    df_BC = df[(df['Group'] == 'B') | (df['Group'] == 'C')]
    df_BC_long = df_BC.melt(id_vars=['Part', 'Group'], value_vars=questions)

    # create new plt figure
    fig = plt.figure()
    # Draw a nested boxplot to show bills by day and time
    ax = sns.violinplot(x="value", y="variable", hue="Group", palette="Set2",
                data=df_BC_long, split=True, scale="count")

    # sns.boxplot(x="value", y="variable", hue="Part", data=df_long)
    # sns.despine(offset=10, trim=True)
    # save the figure
    plt.savefig(f'fig_BC_SUS_combine_part12.png')


# part12_within_subjects(df)

# combine_BC_to_compare_with_linear(df)

compare_BC_without_parts(df)