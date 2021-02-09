# -*- coding: utf-8 -*-
"""
Discriminative power and the most discriminative words per political party 
when applying the models based on Bag of Words (Table 6 and Table 7)

@author: SPraet
"""
#%%
"""
Define file paths
"""
tweets_path = 'PATH_TO_TWEETS' # path to excel file with tweets 


#%%

"""
Import libraries
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

partynames= ['Groen','sp.a','CD&V','Open Vld','NVA', 'VB']

#%%
"""
Define function to find top z most discriminative words using the BoW model
"""

def bow_model(pipe, X, y, z=45):
    """
    Find top z most discriminative words using the BoW model
    
    input:  clf: classification model (LogisticRegression(penalty='L2'))
            X: numpy array with features
            y: list or array with target (1 if from political party, 0 otherwise)
            z: number of issues to consider (default=45)

    output: topz_features: list with top z most discriminative words (unigrams) per political party

    """
    param_search = {'clf__C': [0.001,0.01,0.1,1, 10, 100, 1000]}
    my_cv = TimeSeriesSplit(n_splits=5).split(X)
    gsearch = GridSearchCV(estimator=pipe, cv=my_cv,
                        param_grid=param_search, scoring='roc_auc')
    gsearch.fit(X,y)
    topz = np.argsort(gsearch.best_estimator_.named_steps['clf'].coef_[0])[-z:][::-1]
    topz_features = [gsearch.best_estimator_.named_steps['vect'].get_feature_names()[i] for i in topz]
    
    return topz_features
#%%
"""
Model to predict political party based on BoW
AUC per political party and weighted average AUC (cfr paper Table 7)
"""
tweets = pd.read_excel(tweets_path)

# define features and target
X = tweets['cleaned_ne']
Y = tweets['Party']
indices = list(tweets.index.values)

# define model pipeline   
model = Pipeline([
        ('vect', TfidfVectorizer(analyzer='word', sublinear_tf=True, min_df = 10, max_df=0.90)),
        ('clf', LogisticRegression(penalty='l2', max_iter=7600)),
        ])

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
    
    # grid search 
    param_search = {'clf__C': [0.001,0.01,0.1,1, 10, 100, 1000]}

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
AUC_bow = pd.DataFrame(data={"Party":partynames, 'AUC':auc_parties})

# count tweets per party
counts= tweets["Party"].value_counts().rename_axis('Party').reset_index(name='counts') 
AUC_bow=AUC_bow.merge(counts)

#calculate weighted average AUC
wavg = np.average(AUC_bow["AUC"], weights=AUC_bow["counts"])
AUC_bow.loc[7] = ['All (wavg)',wavg,np.sum(AUC_bow["counts"])]

#%%
"""
Most discriminative words (unigrams) per political party (cfr paper Table 6)
"""
top_words=[]
for partyname in partynames:
    X=np.array(tweets['cleaned_ne'].tolist())
    y = [1 if x == partyname else 0 for x in Y]
    
    top_words.append(bow_model(model,X,y,45))
    
# issues to DataFrame
words_bow = pd.DataFrame(data={"Party":partynames, 'Words':top_words})    
