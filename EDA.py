import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
def inspect_birthday(path) : # 추가변수 후보 1 : 나이에 따른 더미변수 추가
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

def graph_binary(path) : 
    trainset = pd.read_csv(path)
    variable_binary = trainset.columns[trainset.apply(pd.unique, axis=0).apply(len).apply(lambda x : x==2)]
    for variable in variable_binary : 
        if os.path.isdir('binary') == False : 
            os.makedirs('binary')
        trainset_gender = trainset.groupby(['{}'.format(variable)])['credit'].value_counts(normalize=True)
        trainset_gender.plot.bar(grid = True)
        plt.savefig('binary/{}.png'.format(variable))
        plt.clf()

def graph_marriage(path) : 
    trainset = pd.read_csv(path)
    variable_marriage = trainset['family_type'].unique()
    print(variable_marriage)
    if os.path.isdir('family_type') == False : 
        os.makedirs('family_type')
    for variable in variable_marriage : 
        trainset_gender = trainset.groupby(['family_type'])['credit'].value_counts(normalize=True)
        trainset_gender.plot.bar(grid = True)
        plt.savefig('family_type/{}.png'.format(variable))
        plt.clf()


# binary_graph = graph_binaryvariable('train.csv')
birthday = inspect_birthday('train.csv')
graph_binary('train.csv')
graph_marriage('train.csv')



