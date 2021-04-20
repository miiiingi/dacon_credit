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
def sexandcarreality(path) :
    trainset = pd.read_csv(path)
    # print(len(trainset[trainset['credit']==2].loc[trainset['reality']=='Y'])/len(trainset[trainset['reality']=='Y']))
    print(len(trainset[trainset['gender']=='M'])/len(trainset))
sexandcarreality('train.csv')

