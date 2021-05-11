import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os
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

# binary_graph = graph_binaryvariable('train.csv')
birthday = inspect_birthday('train.csv')
graph_binary('train.csv')
graph_marriage('train.csv')
graph_house('train.csv')
graph_income('train.csv')
graph_edu('train.csv')
graph_occyp('train.csv')
graph_child('train.csv')



