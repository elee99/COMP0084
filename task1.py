# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:41:56 2023

@author: elfre
"""

# Packages

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import string
import sys
import collections
import re
import operator
import csv
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import regexp_tokenize
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
from unidecode import unidecode

# Pre-processing

# Validation data
df = pd.read_csv("validation_data.tsv",header=0,sep='\t',low_memory=False)

stop_words = stopwords.words('english')
pattern2 = r"(?:[A-Z]\.)+|\b\w+(?:[-']\w+)*(?:-\d+\w+)*\b|http://[a-zA-Z0-9./]+|(?:[A-Za-z]{1,2}\.)+|[\w\']+|\.\.\.|\?\!|[.?!]|\$?\d+(?:\.\d+)?%?"  

def passage_processing(file):
  # Read in file
  passages = file
  # Use custom pattern to tokenise as before for each passage
  passages['passage'] = [nltk.regexp_tokenize(passages['passage'][i],pattern2) for i in range(len(passages['passage']))]
  passages['passage'] = passages['passage'].replace(r'[^\w\s]+','')
  # Lowercase all words 
  passages['passage'] = passages['passage'].apply(lambda sent: [w.lower() for w in sent])
  # Lemmatizing 
  passages['passage'] = passages['passage'].apply(lambda sent: [lemmatizer.lemmatize(word) for word in sent])
  # Removing unicode characters
  passages['passage'] = passages['passage'].apply(lambda sent: [unidecode(word) for word in sent])
  # Removing stop words
  passages['passage'] = passages['passage'].apply(lambda x: [item for item in x if item not in stop_words])
  # Removing punctuation
  passages['passage'] = passages['passage'].apply(lambda x: [item for item in x if item not in string.punctuation])

  return passages

# Processing passages column
passages = passage_processing(df)
# Saving as file
np.save('val_passages',passages['passage'].tolist(),allow_pickle=True)
# Loading file
passages_val = np.load('val_passages.npy',allow_pickle=True).tolist()


def query_processing(file):
  # Read in file
  passages = file
  # Use custom pattern to tokenise as before for each passage
  passages['queries'] = [nltk.regexp_tokenize(passages['queries'][i],pattern2) for i in range(len(passages['queries']))]
  passages['queries'] = passages['queries'].replace(r'[^\w\s]+','')
  # Lowercase all words 
  passages['queries'] = passages['queries'].apply(lambda sent: [w.lower() for w in sent])
  # Lemmatizing 
  passages['queries'] = passages['queries'].apply(lambda sent: [lemmatizer.lemmatize(word) for word in sent])
  # Removing unicode characters
  passages['queries'] = passages['queries'].apply(lambda sent: [unidecode(word) for word in sent])
  # Removing stop words
  passages['queries'] = passages['queries'].apply(lambda x: [item for item in x if item not in stop_words])
  # Removing punctuation
  passages['queries'] = passages['queries'].apply(lambda x: [item for item in x if item not in string.punctuation])

  return passages

queries_val_processed = query_processing(passages)
# Saving query list 
np.save('val_queries',queries_val_processed['queries'].tolist(),allow_pickle=True)
# Loading query list
queries_val = np.load('val_queries.npy',allow_pickle=True).tolist()

len(queries_val) # 1148 unique queries

# Dropping duplicates
qid_list = df.drop_duplicates(subset='qid')['qid'].tolist()
pid_list = df.drop_duplicates(subset='pid')['pid'].tolist()
queries_drop = queries_val_processed.drop_duplicates(subset='qid')['queries'].tolist()

# Create Inverted Index 

from collections import defaultdict, Counter

# Use pid_full instead? Or drop duplicates from both instead
def inverted_index(pid_list, passages_val):
    # Empty dictionary for inv index
    inverted_index = defaultdict(dict)
    # Loop through all pids in pid_list
    for index, pid in enumerate(pid_list):
        # get corresponding passage for that pid
        passage = passages_val[index]
        # Use Counter to count occurrences of each word in the passage
        word_count = Counter(passage)
        # Insert pid and count into dictionary
        for word, count in word_count.items():
            inverted_index[word][pid] = count
    return inverted_index

inv_index_2 = inverted_index(pid_list,passages_val)
# Saving
np.save('inverted_index',inv_index_2,allow_pickle=True)
inv_index_2 = np.load('inverted_index.npy',allow_pickle=True).item()


# From bm25 we should get top 100 relevant passages for each query (or up to 100)
# these will be ranked in order of relevance
# we then use ndcg to compare ranking to optimal ranking

# Getting average passage length 
passage_length = 0
# Loop through passages
for passage in range(len(passages_val)):
    # Add passage length 
    passage_length += len(passages_val[passage])
av_pl = passage_length/len(passages_val) 
print(av_pl)

# Getting full list of qid and pid
qid_full = df['qid'].tolist()
pid_full = df['pid'].tolist()
queries_full = queries_val_processed['queries'].tolist()

def BM25(qid, k1, k2, b, inv_index, av_pl):
    # Corresponding query
    query = queries_val[qid_full.index(qid)]
    # Get corresponding pids as before
    index_list = [i for i, k in enumerate(qid_full) if k == qid]
    # Assuming relevance information is 0
    ri = 0
    R = 0
    # Total number of passages
    N = len(pid_full)
    # Empty dictionary for bm25 scores
    bm25_list = {}
    # Loop through all corresponding passages
    for i in index_list:
        pid = pid_full[i]  # Get pid
        passage = passages_val[i]  # Get passage
        dl = len(passage)  # Passage length
        K = k1 * ((1 - b) + b * (dl / av_pl))  # Defining constant K
        bm25 = 0
        # Loop through words in query
        # Could also do 'common words' approach
        for word in query:
            qf = query.count(word)  # Number of times word occurs in query
            # Check if word appears in passage
            if word in passage:
                f = passage.count(word)  # Number of times word in passage (tf)
                n = len(inv_index[word])  # Number of passages word occurs in, from inv index
            else:
                f = 0
                n = 0
            # Calculate bm25 score
            bm25 += np.log(((ri + 0.5) / (R - ri + 0.5)) / ((n - ri + 0.5) / (N - n - R + ri + 0.5))) * (
                    k1 + 1) * f * (k2 + 1) * qf / ((K + f) * (k2 + qf))
        # Update main dictionary with bm25 score for that pid
        bm25_list[pid] = bm25
    # Get top 100 (may be less) bm25 scores
    bm25_top100 = dict(sorted(bm25_list.items(), key=lambda x: x[1], reverse=True)[:100])
    return bm25_top100

# BM25 for all queries in format of qid: {pid: bm25 score}
main_bm25 = {}
for qid in qid_list:
    qid_bm25 = BM25(qid,1.2,100,0.75,inv_index_2,av_pl)
    main_bm25.update({qid:qid_bm25})
    
np.save('main_bm25',main_bm25,allow_pickle=True)
main_bm25 = np.load('main_bm25.npy',allow_pickle=True).item()

# Getting relevant passages

def get_relevant(df):
    relev = {}
    not_relev = {}
    relevancy = df['relevancy']

    for ind,qid in enumerate(qid_full):
        pid = pid_full[ind]
        rel = relevancy[ind]
        if rel > 0: 
            if qid not in relev.keys():
                relev[qid] = {pid:ind}
            elif qid in relev.keys():
                new_pid = {pid:ind} 
                relev[qid].update(new_pid)
        else:
            if qid not in not_relev.keys():
                not_relev[qid] = {pid:ind}
            elif qid in not_relev.keys():
                new_pid = {pid:ind}
                not_relev[qid].update(new_pid) 
    return relev,not_relev
relev,not_relev = get_relevant(df)

# Saving
np.save('relev',relev,allow_pickle=True)
relev = np.load('relev.npy',allow_pickle=True)
np.save('not_relev',not_relev,allow_pickle=True)
not_relev = np.load('not_relev.npy',allow_pickle=True)

#%%

def ndcg(bm25_score):
    NDCG = []
    for qid in qid_full:
        rels = relev[qid]  # relevant passages
        # i is pid(s)
        # loop through relevant pid(s)
        none_found = [i not in bm25_score[qid].keys() for i in rels.keys()]
        if none_found == True:
            ndcg_ = 0  
        else:
            opt = {}
            for i in list(rels.keys()):
                # set dcg to 0
                dg = 0
                # if this pid is retrieved by bm25, then we get relevance etc
                # if bm25 does not retrieve pid then ndcg is 0?
                if i in list(bm25_score[qid].keys()):
                    # get rank of that pid (+1 as indexing starts at 0)
                    rank = list(bm25_score[qid].keys()).index(i)+1
                    # get relevancy of that pid
                    # change
                    rel = float(df[rels[i]:rels[i]+1].relevancy)
                    #rel = float(df['relevancy'].iloc[rels.values()])
                    #rel = (df[df['pid']==i]['relevancy']).astype(float)
                    gain = 2**(rel) - 1
                    dg += gain/np.log2(rank+1)
    
                for pid in rels.values():
                    opt[pid] = df.loc[pid, 'relevancy'].tolist()
    
                opt_dcg = 0
                for i, pid in enumerate(sorted(opt.keys(), reverse=True)):
                    rel = opt[pid]
                    rank = i+1
                    gain = 2**(rel) - 1
                    opt_dcg += gain / np.log2(rank+1) 
            ndcg_ = dg/opt_dcg
    
        NDCG.append(ndcg_)
    return NDCG, np.mean(NDCG)
NDCG, mean_NDCG = ndcg(main_bm25) # 0.25

def average_prec():
    # rank
    # relevance
    # precision (relevant/relevant)
    # for example if first doc retrived by bm25 is not relevant and there is 1 rel doc. Precision
    # would be 0/1 
    # divide by total number relevant docs
    # only use precision for relevant retrieved docs
    AP = []
    for qid in qid_full:
        # get relevant pid(s): {pid:position}
        rels = relev[qid]
        retrieved = 0
        prec = 0
        # loop through pid(s)
        for i in list(rels.keys()):
            # if pid(s) retrieved by bm25, continue, if not then AP is 0?
            if i in list(main_bm25[qid].keys()):
                # no of rel docs retrieved / tot docs retrieved
                rank = list(main_bm25[qid].keys()).index(i)+1
                # total docs retrieved is same as rank of current i 
                retrieved += 1
                # need to divide by total relevant docs?
                prec += (retrieved/rank)/len(rels.keys())
        
            else:
                prec += 0
        AP.append(prec) 
    return AP, np.mean(AP)

AP, mean_AP = average_prec()
mean_AP # 0.234
AP    

# End of task 1 #

