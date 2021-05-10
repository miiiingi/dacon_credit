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
def inspect_birthday(path) :
    trainset = pd.read_csv(path)
    trainset['DAYS_BIRTH'] = (-trainset['DAYS_BIRTH'])//365
    group_age_2 = trainset[(trainset['DAYS_BIRTH'] < 30) & (20 <= trainset['DAYS_BIRTH'])]
    group_age_3 = trainset[(trainset['DAYS_BIRTH'] < 40) & (30 <= trainset['DAYS_BIRTH'])]
    group_age_4 = trainset[(trainset['DAYS_BIRTH'] < 50) & (40 <= trainset['DAYS_BIRTH'])]
    group_age_5 = trainset[(trainset['DAYS_BIRTH'] < 60) & (50 <= trainset['DAYS_BIRTH'])]
    group_age_6 = trainset[(trainset['DAYS_BIRTH'] < 70) & (60 <= trainset['DAYS_BIRTH'])]
    agelist = [[] for _ in range(5)]
    for ind, age in enumerate([group_age_2, group_age_3, group_age_4, group_age_5, group_age_6]) : 
        for credit in [x for x in range(3)] : 
            agelist[ind].append((sum(age['credit'] == credit)/len(age)))
    return agelist
def inspect_binary(path) : 
    trainset = pd.read_csv(path)
    variable_binary = trainset.columns[trainset.apply(pd.unique, axis=0).apply(len).apply(lambda x : x==2)]
    list_value = [[] for _ in range(len(variable_binary))]
    for ind, variable in enumerate(variable_binary) : 
        for value in trainset[variable].unique() : 
            for credit in range(3) : 
                variable_value = (trainset[variable] == value)
                variable_credit = (trainset['credit'] == credit)
                list_value[ind].append((value,sum(variable_value & variable_credit)/sum(variable_value) ))
    return list_value 
# binary_graph = graph_binaryvariable('train.csv')
# birthday = inspect_birthday('train.csv')
binary_inspect = inspect_binary('train.csv')
print(binary_inspect)



