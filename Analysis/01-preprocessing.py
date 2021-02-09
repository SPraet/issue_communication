# -*- coding: utf-8 -*-
"""
Preprocessing of tweets (see section 3.2 in paper)
- replace named entities, mentions, numbers, and urls
- remove punctuation and stopwords (Dutch, French, and English)
- normalize (reduce to word lemma)

@author: SPraet
"""
#%%
"""
Define file paths
"""
tweets_path = '../tweets.xlsx' # path to excel file with tweets 
output_path = 'tweets_cleaned.xlsx' # path to excel output file with preprocessed tweets

#%%

"""
Import libraries
"""

#!pip install spacy # install spacy
#!python -m spacy download nl_core_news_sm # download trained spacy pipeline for Dutch
import pandas as pd
import nltk  # see https://www.nltk.org/
nltk.download('stopwords')
from nltk.corpus import stopwords 
from pattern.nl import parse # see https://github.com/clips/pattern
import string
import spacy # see https://spacy.io/models/nl
nlp = spacy.load("nl_core_news_sm")

#%%
"""
Define functions to clean documents
- replace named entities, mentions, numbers, and urls
- remove punctuation and stopwords (Dutch, French, and English)
- normalize (reduce to word lemma)
"""

# define stopwords and punctuation list
stop = set(stopwords.words('dutch'))
stop.update(set(stopwords.words('french')))
stop.update(set(stopwords.words('english')))
stop.update(['wij', 'jij','je','gij','ge', 'ie', 'amp', 'neen','wa','ni']) # add stopwords to list
exclude = set(string.punctuation)

# functions to replace Named Entities
def replace_NE(token):
    if token.ent_iob in [1,3]:
        return 'zzz' + token.ent_type_ + ' ' 
    else:
        return str(token)+ ' ' 

def redact_NE(nlp_doc):
    #for ent in nlp_doc.ents:
    #    ent.merge()
    tokens = map(replace_NE, nlp_doc)
    return ''.join(tokens)

# function to clean documets (tweets)
def clean(doc):
    https_free = " ".join([i if not i.startswith('http') else 'zzzurl' for i in doc.split()])
    mention_free = " ".join([j if not j.startswith('@') else 'zzzmention' for j in https_free.split()])
    survey_doc = nlp(mention_free)
    doc_ne = redact_NE(survey_doc)
    punc_free = ''.join(ch for ch in doc_ne.lower() if ch not in exclude)
    stop_free = " ".join([x for x in punc_free.split() if x not in stop])
    number_free = ''.join([k if not k.isdigit() else 'zzznumber' for k in stop_free ])
    normalized = " ".join([(parse(word, lemmata=True).split())[0][0][4] for word in number_free.split()])
    return normalized

#%%
"""
Read excel with tweets to pandas DataFrame and add column with cleaned text
"""
tweets = pd.read_excel(tweets_path)
doc_clean = [clean(doc) for doc in tweets['text']]
tweets['cleaned_ne'] = doc_clean

tweets.to_excel(output_path)