# -*- coding: utf-8 -*-
"""
Load json file with hydrated tweets line by line and append 
tweet_id, screen_name, text,created_at,  hashtags, mentions, 
and retweeted_user to list. Convert list to pandas dataframe 
and store as excel (or csv) file

@author: SPraet
"""
#%%

import pandas as pd
import json

#%%
"""
Add file path to json file and output file
"""
json_path = "test.json" # add filepath to json file with hydrated tweets
output_path = 'tweets.xlsx' # adjust filepath to output file

#%%
"""
Function to read json file with tweets line by line and append to list
"""

def json_to_list(file):
    with open(file) as f: 
        tweet_list=[]
        for line in f:
            try:
                tweet = json.loads(line)
            except:
                pass
            if 'user' in tweet.keys():
               if 'retweeted_status' in tweet.keys(): # only in case of retweet
                   retweeted_screen_name=tweet['retweeted_status']['user']['screen_name']
                   if 'extended_tweet' in tweet['retweeted_status'].keys():
                       text= tweet['retweeted_status']['extended_tweet']['full_text']
                   else:
                       text= tweet['retweeted_status']['text']
               else:
                   retweeted_screen_name= '/'
                   if 'extended_tweet' in tweet.keys():
                       text= tweet['extended_tweet']['full_text']
                   else:
                       text= tweet['text']
                       
               tweet_list.append([tweet['id'], tweet['user']['screen_name'], text, 
                                  tweet['created_at'], [hashtag['text'] for hashtag in tweet['entities']['hashtags']],
                                  [mention['screen_name'] for mention in tweet['entities']['user_mentions']], 
                                  retweeted_screen_name])
    return tweet_list  
              
#%%
"""
Apply function, convert list to pandas DataFrame and store in Excel
"""
tweet_list = json_to_list(json_path)
columns=['tweet_id','screen_name', 'text','created_at', 'hashtags', 'mentions','retweeted_user']   
df_tweets = pd.DataFrame(data=tweet_list, columns=columns)
df_tweets.to_excel(output_path, encoding='utf-8')
