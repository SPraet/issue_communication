# -*- coding: utf-8 -*-
"""
Discriminative power and the most discriminative words per political party 
when applying the models based topic modeling (NMF) (Table C2 and Table 7)


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
from sklearn.decomposition import NMF

partynames= ['Groen','sp.a','CD&V','Open Vld','NVA', 'VB']

#%%
"""
Define functions
"""
def display_topics(topics, feature_names, w):
    """
    Display topics extracted by topic model by top w words with highest weight 
    
    input:  topics: topics to represent (eg. nmf.components_)
            feature_names: feature names of applied vectorizer (eg. vect.get_feature_names())
            w: number of words per topic 

    output: displayed_topics: list with strings containing topic name and top w words

    """
    displayed_topics=[]
    for topic_idx, topic in enumerate(topics):
        displayed_topics.append("Topic %d:" % (topic_idx) + 
                                " ".join([feature_names[i] for i in topic.argsort()[:-w - 1:-1]]))
    return displayed_topics


def topics_model(pipe, X, y, z=3, w=15):
    """
    Find top z most discriminative words using the BoW model
    
    input:  clf: classification model (LogisticRegression(penalty='L2'))
            X: numpy array with features
            y: list or array with target (1 if from political party, 0 otherwise)
            z: number of top NMF topics to consider (default=3)
            w: number of words per NMF topic (default =15)

    output: topz_features: list with top z most discriminative words (unigrams) per political party

    """
    param_search = {'nmf__n_components':[350],'clf__C': [0.001,0.01,0.1,1, 10, 100, 1000]}
    my_cv = TimeSeriesSplit(n_splits=5).split(X)
    gsearch = GridSearchCV(estimator=pipe, cv=my_cv,
                        param_grid=param_search, scoring='roc_auc')
    gsearch.fit(X,y)
    topz = np.argsort(gsearch.best_estimator_.named_steps['clf'].coef_[0])[-z:][::-1]
    topz_features = [gsearch.best_estimator_.named_steps['nmf'].components_[i] for i in topz]
    feature_names=gsearch.best_estimator_.named_steps['vect'].get_feature_names()
    displayed_tops = display_topics(topz_features, feature_names, w)
    
    return displayed_tops
        
#%%
"""
Model to predict political party based on topic modeling
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
         ('nmf', NMF(random_state=1, alpha=.1, l1_ratio=.5, init='nndsvd')),
        ('clf', LogisticRegression(penalty='l2', max_iter=7600)),
        ])

# out-of-time 20% test data
trainval_idx, test_idx  = train_test_split(indices, test_size=0.20, shuffle = False, stratify = None)
Y_trainval, Y_test = Y[trainval_idx], Y[test_idx]        
X_trainval, X_test= np.array(X.iloc[trainval_idx].tolist()), np.array(X.iloc[test_idx].tolist())

# find optimal number of topics (k)
auc_parties = []
best_C = []
for partyname in partynames:

    y_trainval = [1 if x == partyname else 0 for x in Y_trainval]
    y_test = [1 if x == partyname else 0 for x in Y_test]
    
    # grid search to find optimal number of topics (k) for constant C
    param_search = {'nmf__n_components': list(np.arange(100,450,50))}
    my_cv = TimeSeriesSplit(n_splits=5).split(X_trainval)
    gsearch = GridSearchCV(estimator=model, cv=my_cv,
                        param_grid=param_search, scoring='roc_auc')
    gsearch.fit(X_trainval, y_trainval)
    
    # grids search find optimal C for optimal k from previous step
    param_search = {'nmf__n_components': [gsearch.best_params_['nmf__n_components']], 'clf__C': [0.001,0.01,0.1,1, 10, 100, 1000]}
    my_cv = TimeSeriesSplit(n_splits=5).split(X_trainval)
    gsearch = GridSearchCV(estimator=model, cv=my_cv,
                        param_grid=param_search, scoring='roc_auc')
    gsearch.fit(X_trainval, y_trainval)

    # prediction on test set
    pred = gsearch.predict_proba(X_test) #Call predict_proba on the estimator with the best found parameters.
    score=roc_auc_score(y_test, pred[:,1])

    auc_parties.append(score)

# AUC to DataFrame
AUC_topic_model = pd.DataFrame(data={"Party":partynames, 'AUC':auc_parties})

# count tweets per party
counts= tweets["Party"].value_counts().rename_axis('Party').reset_index(name='counts') 
AUC_topic_model=AUC_topic_model.merge(counts)

#calculate weighted average AUC
wavg = np.average(AUC_topic_model["AUC"], weights=AUC_topic_model["counts"])
AUC_topic_model.loc[7] = ['All (wavg)',wavg,np.sum(AUC_topic_model["counts"])]

#%%
"""
Most discriminative NMF topics per political party (cfr paper Table C2)
"""
top_topics=[]
for partyname in partynames:
    X=np.array(tweets['cleaned_ne'].tolist())
    y = [1 if x == partyname else 0 for x in Y]
    
    top_topics.append(topics_model(model,X,y,3,15))

# topics to DataFrame
topics_NMF = pd.DataFrame(data={"Party":partynames, 'Topics':top_topics})    
