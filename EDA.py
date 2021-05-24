from part import make_dataset
import pandas as pd 
import matplotlib.pyplot as plt
import os
import numpy as np 
import copy
def inspect_birthday(path) : # 추가변수 후보 1 : 나이에 따른 더미변수 추가 >> 더 생각해볼 것 : 365를 기준으로 몫만 취하는 것이 맞는가 ?? > 몫만 취하게 되면 364일은 아무것도 아니게 되니까
    # 그리고 20, 30, 40 대 등으로 나누는 것이 맞는가 ?? >> 기준이 어디인지 모르니까 다르게 나눠보자 !
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

def graph_marriage(path) : # 추가변수 후보 2 : 가족형태에 따른 더미변수 추가 
    trainset = pd.read_csv(path)
    if os.path.isdir('family_type') == False : 
        os.makedirs('family_type')
    trainset_marriage = trainset.groupby(['family_type'])['credit'].value_counts(normalize=True)
    trainset_marriage.plot.bar(grid = True)
    plt.tight_layout()
    plt.savefig('family_type/marriage.png')

def graph_house(path) : # 추가변수 후보 3 : 주거형태에 따른 더미변수 추가 
    trainset = pd.read_csv(path)
    if os.path.isdir('house_type') == False : 
        os.makedirs('house_type')
    trainset_house = trainset.groupby(['house_type'])['credit'].value_counts(normalize=True)
    trainset_house.plot.bar(grid = True)
    plt.tight_layout()
    plt.savefig('house_type/house.png')

def graph_income(path) : # 추가변수 후보 4 : 수입형태에 따른 더미변수 추가  / working ??
    trainset = pd.read_csv(path)
    if os.path.isdir('income_type') == False : 
        os.makedirs('income_type')
    trainset_income = trainset.groupby(['income_type'])['credit'].value_counts(normalize=True)
    trainset_income.plot.bar(grid = True)
    plt.tight_layout()
    plt.savefig('income_type/income_type.png')

def graph_edu(path) : # 추가변수 후보 5 : 교육정도에 따른 더미변수 추가 / lower secondary ?? 
    trainset = pd.read_csv(path)
    if os.path.isdir('edu_type') == False : 
        os.makedirs('edu_type')
    trainset_edu = trainset.groupby(['edu_type'])['credit'].value_counts(normalize=True)
    trainset_edu.plot.bar(grid = True)
    plt.tight_layout()
    plt.savefig('edu_type/edu_type.png')

def graph_occyp(path) : # 추가변수 후보 6 : 고용형태에 따른 더미변수 추가
    trainset = pd.read_csv(path)
    if os.path.isdir('occyp_type') == False : 
        os.makedirs('occyp_type')
    trainset_occyp = trainset.groupby(['occyp_type'])['credit'].value_counts(normalize=True)
    trainset_occyp.plot.bar(grid = True)
    plt.tight_layout()
    plt.savefig('occyp_type/occyp_type.png')

def graph_child(path) : # 추가변수 후보 7 : 아이수에 따른 분석 >> 가족형태랑 다시 조사해보기 (한 명 이었을 때, 상대적으로 낮은데, 이것이 과부랑 관련?)
    trainset = pd.read_csv(path)
    if os.path.isdir('child_num') == False : 
        os.makedirs('child_num')
    trainset_child = trainset.groupby(['child_num'])['credit'].value_counts(normalize=True)
    trainset_child.plot.bar(grid = True)
    plt.tight_layout()
    plt.savefig('child_num/child_num.png')

def graph_days_employed2(path) : # 추가변수 후보 8 : 고용형태 >>  현재 은퇴한 사람과 은퇴하지 않은 사람간의 차이가 있는지 보려고 만든 변수 그렇게 큰 차이는 없음..
    trainset = pd.read_csv(path)
    if os.path.isdir('days_employed') == False : 
        os.makedirs('days_employed')
    trainset.loc[trainset['DAYS_EMPLOYED'] < 0, 'DAYS_EMPLOYED'] = -trainset.loc[trainset['DAYS_EMPLOYED'] < 0, 'DAYS_EMPLOYED']
    trainset.loc[trainset['DAYS_EMPLOYED'] == 365243, 'DAYS_EMPLOYED'] = 0
    x = copy.deepcopy(trainset)
    lower = np.percentile(x['DAYS_EMPLOYED'], 0)  
    for p in range(10, 110, 10) : 
        upper = np.percentile(x['DAYS_EMPLOYED'], p) 
        trainset.loc[((lower <= trainset['DAYS_EMPLOYED']) & (trainset['DAYS_EMPLOYED'] <= upper)), 'DAYS_EMPLOYED'] = p // 10
        lower = upper
    trainset_days_employed = trainset.groupby(['DAYS_EMPLOYED'])['credit'].value_counts(normalize=True)
    trainset_days_employed.plot.bar(grid = True)
    plt.tight_layout()
    plt.savefig('days_employed/days_employed2.png')

def graph_occpy(path) : # 추가변수 후보 9 고용형태와 직업 >> 고용되어 있지 않지만, 은퇴한 사람과 고용되어 있지 않지만, 은퇴하지 않은 사람에 대한 변수 아주 미묘하지만 차이가 있음..
    trainset = pd.read_csv(path)
    if os.path.isdir('days_occpy') == False : 
        os.makedirs('days_occpy')
    trainset.loc[trainset['occyp_type'].isnull() & (trainset['DAYS_EMPLOYED'] == 365243), 'occyp_type'] = 1
    trainset.loc[trainset['occyp_type'].isnull() & (trainset['DAYS_EMPLOYED'] != 365243), 'occyp_type'] = 0
    trainset_days_employed = trainset.groupby(['occyp_type'])['credit'].value_counts(normalize=True)
    trainset_days_employed.plot.bar(grid = True)
    plt.tight_layout()
    plt.savefig('days_occpy/days_occpy.png')

def graph_begin_month(path) : # 추가변수 후보 10 : 신용카드 발급기간 >> 신용카드 발급을 받은 기간이 얼마 인지에 따라 아주 미묘한 차이가 있음 
    trainset = pd.read_csv(path)
    if os.path.isdir('begin_month') == False : 
        os.makedirs('begin_month')
    x = copy.deepcopy(trainset)
    lower = np.percentile(x['begin_month'], 0)  
    for p in range(10, 110, 10) : 
        upper = np.percentile(x['begin_month'], p) 
        print('{}~{}'.format(lower, upper))
        trainset.loc[((lower <= trainset['begin_month']) & (trainset['begin_month'] <= upper)), 'begin_month'] = p // 10
        lower = upper
    trainset_days_employed = trainset.groupby(['begin_month'])['credit'].value_counts(normalize=True)
    trainset_days_employed.plot.bar(grid = True)
    plt.tight_layout()
    plt.savefig('begin_month/begin_month.png')

def make_tsne(dataset_x, dataset_y) : 
    from sklearn.manifold import TSNE
    colors = ['#476A2A', '#7851B8', '#BD3430']
    tsne = TSNE(random_state=722)
    result_tsne = tsne.fit_transform(dataset_x)
    plt.figure(figsize=(10, 10))
    plt.xlim(result_tsne[:, 0].min(), result_tsne[:, 0].max() + 1)
    plt.ylim(result_tsne[:, 1].min(), result_tsne[:, 1].max() + 1)
    for i in range(len(dataset_x)) : 
        plt.text(result_tsne[i, 0], result_tsne[i, 1], str(dataset_y[i]), color = colors[dataset_y[i]], fontdict = {'weight' : 'bold', 'size' : 9})
    plt.xlabel('character0')
    plt.ylabel('character1')
    plt.savefig('tsne.png')

# binary_graph = graph_binaryvariable('train.csv')
# birthday = inspect_birthday('train.csv')
# graph_binary('train.csv')
# graph_marriage('train.csv')
# graph_house('train.csv')
# graph_income('train.csv')
# graph_edu('train.csv')
# graph_occyp('train.csv')
# graph_child('train.csv')
# graph_days_employed2('train.csv')
# graph_occpy('train.csv')
# graph_begin_month('train.csv')
data_x, data_y = make_dataset('train')
make_tsne(data_x, data_y)




