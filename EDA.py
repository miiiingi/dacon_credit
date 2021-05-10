import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
def graph_binaryvariable(path) : 
    trainset = pd.read_csv(path)
    del trainset['index']
    variable_binary = trainset.columns[trainset.apply(pd.unique, axis=0).apply(len).apply(lambda x : x==2)]
    fig, axes = plt.subplots(1, len(variable_binary), figsize = (15, 3))
    for ind, variable in enumerate(variable_binary) :
        legendlist = [] 
        for category in trainset[variable].unique() : 
            if type(category) == str : 
                print(category)
                sns.distplot(trainset['credit'][trainset['{}'.format(variable)]=='{}'.format(category)], ax = axes[ind])
            else : 
                sns.distplot(trainset['credit'][trainset['{}'.format(variable)]==category], ax = axes[ind])
            legendlist.append(category)
        fig.legend(labels=legendlist)
        axes[ind].set_title('{}'.format(variable))
    plt.show()
# 1. graph binary variable 
# graph_binaryvariable('train.csv')
def inspect_birthday(path) :
    trainset = pd.read_csv(path)
    trainset['DAYS_BIRTH'] = (-trainset['DAYS_BIRTH'])//365
    group_age_2 = trainset[(trainset['DAYS_BIRTH'] < 30) & (20 <= trainset['DAYS_BIRTH'])]
    group_age_3 = trainset[(trainset['DAYS_BIRTH'] < 40) & (30 <= trainset['DAYS_BIRTH'])]
    group_age_4 = trainset[(trainset['DAYS_BIRTH'] < 50) & (40 <= trainset['DAYS_BIRTH'])]
    group_age_5 = trainset[(trainset['DAYS_BIRTH'] < 60) & (50 <= trainset['DAYS_BIRTH'])]
    group_age_6 = trainset[(trainset['DAYS_BIRTH'] < 70) & (60 <= trainset['DAYS_BIRTH'])]
    agelist = [[] for _ in range(5)]
    for age in agelist : 
        for credit in [x for x in range(3)] : 
            print(sum(group_age_6['credit'] == credit)/len(group_age_6))

inspect_birthday('train.csv')

