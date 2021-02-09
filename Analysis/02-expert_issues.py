# -*- coding: utf-8 -*-
"""
Discriminative power and the most discriminative issues per political party 
when applying the models based on expert issues (Table 6 and Table 7)

@author: SPraet
"""

#%%
"""
Define file paths
"""
dict_path= '20140718_dutchdictionary_v2.lcd' # path to dictionary
tweets_path = 'tweets_cleaned.xlsx' # path to excel file with tweets 


#%%

"""
Import libraries
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

partynames= ['Groen','sp.a','CD&V','Open Vld','NVA', 'VB']

#%%
"""
Load dictionary
"""
import xml.etree.ElementTree as et
def load_issues(fname):
    issues = {}
    tree = et.parse(fname)
    root = tree.getroot()
    for child in root:
        issue = child.attrib['name']
        queries = [query.attrib['name'].lower() for query in child]
        issues[issue] = queries
    return issues

issues = load_issues(dict_path)

del issues['t24']
del issues['t29']

issues_list = ['Macroeconomics', 'Human rights', 'Health',
'Agriculture', 'Labor and employment', 'Immigration', 'Education', 'Environment', 'Energy', 'Transportation',
'Law and crime', 'Social welfare', 'Community development', 'Banking and finance','Defense', 'Science and technology',
'Foreign trade', 'International affairs','Government operations', 'Public lands and water', 'Culture and arts']

#%%
"""
Function to transform tweet to expert issues using dictionary and 
to train classification model to predict political party based on expert issues
"""

def to_issues_vector(text, issues, x=1, norm=False):
    """
    input: text: the text of tweets as string
           issues: issues dictionary
           x: only assign issue if more than x dictionary words appear in text, default = 1 
           norm: if norm is true the word count is divided by the length of the text. default = False

    output: issues_vec: list of values with len = len(topcis). 0 when issue does not appear in text, 1 or int when issue appears

    """
    text = text.lower()
    length = len(text.split(' '))
    issue_list = []  
    issues_vec = []
    for issue, words in issues.items(): 
        issue_list.append(issue)
        count = 0
        for word in words:
            count += text.count(word)
        if count > x: 
            if norm==False:
                issues_vec.append(1) # 1 if issue appears, 0 otherwise 
            if norm==True:
                issues_vec.append(count/length) # number of words divided by length of tweet 
        else:
            issues_vec.append(0)

    return issues_vec


def exp_issue_model(clf, X, y, z=3):
    """
    Find top z most discriminative issues using the expert issue model
    
    input:  clf: classification model (LogisticRegression(penalty='L2'))
            X: numpy array with features
            y: list or array with target (1 if from political party, 0 otherwise)
            z: number of issues to consider (default=3)

    output: topz_features: list with top z most discriminative features per political party

    """
    param_search = {'C': [0.001,0.01,0.1,1, 10, 100, 1000]}
    my_cv = TimeSeriesSplit(n_splits=5).split(X)
    gsearch = GridSearchCV(estimator=clf, cv=my_cv,
                        param_grid=param_search, scoring='roc_auc')
    gsearch.fit(X,y)
    coef = gsearch.best_estimator_.coef_[0]
    topz = np.argsort(coef)[-z:][::-1]
    topz_features = [issues_list[i] for i in topz]
    
    return topz_features

#%%
"""
Read excel with tweets to pandas DataFrame and add column with issues vector
"""

tweets = pd.read_excel(tweets_path)
tweets['expert_issues'] = tweets['text'].apply(lambda x: to_issues_vector(x, issues, 0, norm=False))

#%%
"""
Model to predict political party based on expert issues
AUC per political party and weighted average AUC (cfr paper Table 7)
"""

# define features and target
X = tweets['expert_issues']
Y = tweets['Party']
indices = list(tweets.index.values)

# out-of-time 20% test data
trainval_idx, test_idx  = train_test_split(indices, test_size=0.20, shuffle = False, stratify = None)
Y_trainval, Y_test = Y[trainval_idx], Y[test_idx]        
X_trainval, X_test= np.array(X.iloc[trainval_idx].tolist()), np.array(X.iloc[test_idx].tolist())

# AUC per political party
auc_parties = []
best_C = []
for partyname in partynames:

    y_trainval = [1 if x == partyname else 0 for x in Y_trainval]
    y_test = [1 if x == partyname else 0 for x in Y_test]

    # grid search LR
    param_search = {'C': [0.001,0.01,0.1,1, 10, 100, 1000]}
    model = LogisticRegression(penalty='l2', max_iter=7600)
    my_cv = TimeSeriesSplit(n_splits=5).split(X_trainval)
    gsearch = GridSearchCV(estimator=model, cv=my_cv,
                        param_grid=param_search, scoring='roc_auc')
    gsearch.fit(X_trainval, y_trainval)
    best_C.append(gsearch.best_params_)
    
    # prediction on test set
    pred = gsearch.predict_proba(X_test) #Call predict_proba on the estimator with the best found parameters.
    score=roc_auc_score(y_test, pred[:,1])

    auc_parties.append(score)

# AUC to DataFrame
AUC_expert_issues = pd.DataFrame(data={"Party":partynames, 'AUC':auc_parties})

# count tweets per party
counts= tweets["Party"].value_counts().rename_axis('Party').reset_index(name='counts') 
AUC_expert_issues=AUC_expert_issues.merge(counts)

#calculate weighted average AUC
wavg = np.average(AUC_expert_issues["AUC"], weights=AUC_expert_issues["counts"])
AUC_expert_issues.loc[7] = ['All (wavg)',wavg,np.sum(AUC_expert_issues["counts"])]

#%%
"""
Most discriminative issues per political party (cfr paper Table 6)
"""
top_issues=[]
for partyname in partynames:
    X=np.array(tweets['expert_issues'].tolist())
    y = [1 if x == partyname else 0 for x in Y]
    
    top_issues.append(exp_issue_model(LogisticRegression(penalty='l2'),X,y,3))
    
# issues to DataFrame
issues_expert_issues = pd.DataFrame(data={"Party":partynames, 'Issues':top_issues})    
