# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 11:14:25 2023

@author: elfre
"""

import os
import pickle
import xgboost as xgb
import numpy as np
import pandas as pd

X_train = np.load('X_train.npy', allow_pickle=True)
y_train = np.load('y_train.npy', allow_pickle=True)

# Including passage and query length
X_train_2 = np.load('X_train_2.npy', allow_pickle=True)
X_test_2 = np.load('X_test_2.npy', allow_pickle=True)

merged_df = pd.read_csv('saved_df.csv', header=0)

groups = merged_df.groupby('qid').size().to_frame('size')['size'].to_numpy()

"""THIS CODE WAS ORIGINALLY RUN IN GOOGLE COLAB"""

# Params for tuning 
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'colsample_bytree': [0.5, 0.75, 1.0],
}

# 81 total models
total_models = []
for i in range(len(param_grid['n_estimators'])):
    for j in range(len(param_grid['learning_rate'])):
        for k in range(len(param_grid['max_depth'])):
            for l in range(len(param_grid['colsample_bytree'])):
                total_models.append(xgb.XGBRanker(
                    objective='rank:pairwise',
                    tree_method='gpu_hist',
                    n_estimators=param_grid['n_estimators'][i],
                    learning_rate=param_grid['learning_rate'][j],
                    max_depth=param_grid['max_depth'][k],
                    colsample_bytree=param_grid['colsample_bytree'][l],
                    booster='gbtree'))
num_models = len(total_models)

# fitting all models to training data
models = []
i = 1
for model in total_models:
    models.append(model.fit(X_train_2, y_train, group=groups, verbose=0))
    i += 1
    
"""END OF COLAB CODE"""

#%%

merged_val_df = pd.read_csv('saved_val_df.csv', header=0)
#validation_df = df
validation_df = pd.read_csv("validation_data.tsv",
                            header=0, sep='\t', low_memory=False)
validation_df['query length'] = validation_df['queries'].apply(len)
validation_df['passage length'] = validation_df['passage'].apply(len)

query_lengths = validation_df['query length'].values
passage_lengths = validation_df['passage length'].values

pid_list_full = list(merged_val_df.pid.values)
qid_list_full = list(merged_val_df.qid.values)

qid_list = list(set(qid_list_full))
pid_list = list(set(pid_list_full))

def str_to_array(num_str):
    num_str = num_str.strip('[]').split(',')
    num_array = np.array([float(x) for x in num_str])
    return num_array

merged_val_df['passage embedding'] = merged_val_df['passage embedding'].apply(
    str_to_array)
merged_val_df['query embedding'] = merged_val_df['query embedding'].apply(
    str_to_array)

query_embeddings1 = merged_val_df['query embedding'].values
passage_embeddings1 = merged_val_df['passage embedding'].values

# Getting scores for LM models

# For one qid
def test_data(df, qid, query_embeddings, passage_embeddings, query_lengths, passage_lengths, model):
    index_q = qid_list_full.index(qid)
    query_embedding = query_embeddings[index_q]

    ind_list = np.where(qid_list_full == qid)[0]
    p_embeddings = passage_embeddings1[ind_list]
    qlengths = query_lengths[ind_list]
    plengths = passage_lengths[ind_list]
    pid_list = []
    scores_dict = {}

    total_arr = []
    for i, embedding in enumerate(p_embeddings):
        comb = np.hstack(
            (query_embedding, embedding, qlengths[i], plengths[i]))
        total_arr.append(comb)
        pid_list.append(pid_list_full[ind_list[i]])

    total_arr = np.array(total_arr)
    scores = model.predict(total_arr)

    for j in range(len(pid_list)):
        scores_dict.update({pid_list[j]: float(scores[j])})

    top = dict(sorted(scores_dict.items(),
               key=lambda x: x[1], reverse=True)[:100])

    return top

# get scores for each model, save them, compute NDCG and AP
# pick best model

# Naming models
for i, model in enumerate(models):
    model.name = f"XGBRanker_{i+1}"

# For all qids, one model
def get_lm_scores(model, qid_list, merged_val_df, query_embeddings, passage_embeddings, query_lengths, passage_lengths):
    lm_score = {}
    for qid in qid_list:
        top_score = test_data(merged_val_df, qid, query_embeddings1,
                              passage_embeddings1, query_lengths, passage_lengths, model)
        # if len(top_score) == 100:
        lm_score[qid] = top_score
    print(f"Completed {model.name}")
    return lm_score

# For all models, all qids 
all_lm_scores = {}
for model in models:
    lm_scores = get_lm_scores(model, qid_list, merged_val_df, query_embeddings1,
                              passage_embeddings1, query_lengths, passage_lengths)
    all_lm_scores[model.name] = lm_scores

np.save('model_scores_lm', all_lm_scores, allow_pickle=True)

# %%
qid_full = validation_df['qid'].tolist()
pid_full = validation_df['pid'].tolist()

def get_relevant(df):
    relev = {}
    not_relev = {}
    relevancy = df['relevancy']
    for ind, qid in enumerate(qid_full):
        pid = pid_full[ind]
        rel = relevancy[ind]
        if rel > 0:
            if qid not in relev.keys():
                relev[qid] = {pid: ind}
            elif qid in relev.keys():
                new_pid = {pid: ind}
                relev[qid].update(new_pid)
        else:
            if qid not in not_relev.keys():
                not_relev[qid] = {pid: ind}
            elif qid in not_relev.keys():
                new_pid = {pid: ind}
                not_relev[qid].update(new_pid)
    return relev, not_relev

relev, not_relev = get_relevant(validation_df)

def average_prec(lm_score):
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
            if i in list(lm_score[qid].keys()):
                # no of rel docs retrieved / tot docs retrieved
                rank = list(lm_score[qid].keys()).index(i)+1
                # total docs retrieved is same as rank of current i
                retrieved += 1
                # need to divide by total relevant docs?
                prec += (retrieved/rank)/len(rels.keys())

            else:
                prec += 0
        AP.append(prec)
    return AP, np.mean(AP)

def ndcg(lm_score):
    NDCG = []
    for qid in qid_full:
        rels = relev[qid]  # relevant passages
        # i is pid(s)
        # loop through relevant pid(s)
        none_found = [i not in lm_score[qid].keys() for i in rels.keys()]
        if none_found == True:
            ndcg_ = 0  
        else:
            opt = {}
            for i in list(rels.keys()):
                # set dcg to 0
                dg = 0
                
                if i in list(lm_score[qid].keys()):
                    # get rank of that pid (+1 as indexing starts at 0)
                    rank = list(lm_score[qid].keys()).index(i)+1
                    # get relevancy of that pid
                    # change
                    rel = float(validation_df[rels[i]:rels[i]+1].relevancy)
                    #rel = float(df['relevancy'].iloc[rels.values()])
                    #rel = (df[df['pid']==i]['relevancy']).astype(float)
                    gain = 2**(rel) - 1
                    dg += gain/np.log2(rank+1)
    
                for pid in rels.values():
                    opt[pid] = validation_df.loc[pid, 'relevancy'].tolist()
    
                opt_dcg = 0
                for i, pid in enumerate(sorted(opt.keys(), reverse=True)):
                    rel = opt[pid]
                    rank = i+1
                    gain = 2**(rel) - 1
                    opt_dcg += gain / np.log2(rank+1) 
            ndcg_ = dg/opt_dcg
    
        NDCG.append(ndcg_)
    return NDCG, np.mean(NDCG)


NDCG, mean_NDCG = ndcg()

# %%
# Getting all mAP scores for all models

all_average_precs = {}
for score in list(all_lm_scores):
    AP, mean_AP = average_prec(all_lm_scores[score])
    all_average_precs[score] = {'mean_AP': mean_AP}

np.save('average_precs_lm', all_average_precs, allow_pickle=True)

# Sorting in decreasing order
all_av_precs = dict(
    sorted(all_average_precs.items(), key=lambda item: item[1]['mean_AP'], reverse=True))

np.save('final_lm_ap_scores', all_av_precs, allow_pickle=True)

#%%
# Getting all mNDCG scores for all models

all_ndcg = {}
for score in list(all_lm_scores):
    NDCG, mean_NDCG = ndcg(all_lm_scores[score])
    all_average_precs[score] = {'mean_NDCG': mean_NDCG}

all_ndcg = dict(sorted(all_ndcg.items(), key=lambda item: item[1]['mean_NDCG'], reverse=True))

np.save('final_lm_ndcg_scores', all_ndcg, allow_pickle=True)

#%%
# Re-ranking candidate passages 

# top scoring model is model 17
top_scoring_model = models[16]

test_queries_av = np.load('test_queries_av.npy',allow_pickle=True).item()
candidate_p_av = np.load('candidate_p_av.npy',allow_pickle=True).item()

candidate_df = pd.read_csv('candidate_df.csv',header=0)

candidate = pd.read_csv("candidate-passages-top1000.tsv",sep='\t',names=['qid','pid','queries','passage'])
test_query = pd.read_csv("test-queries.tsv",sep='\t',names=['qid','queries'])

qid_candidate_full = np.load('candidate_qids.npy',allow_pickle=True)
pid_candidate_full = np.load('candidate_pids.npy',allow_pickle=True)
query_lengths = np.load('candidate_qlength.npy',allow_pickle=True)
passage_lengths = np.load('candidate_plength.npy',allow_pickle=True)
tes_qid = np.load('test_qids.npy',allow_pickle=True)

def rank_candidate(df, qid, query_embeddings, passage_embeddings, query_lengths, passage_lengths, model):
    index_list = [i for i,k in enumerate(qid_candidate_full) if k == qid]
    # is this right? dict form not list
    query_embedding = test_queries_av[qid]
    
    pid_list = []
    scores_dict = {}
    total_arr = []
    
    for i in index_list:
        p_embedding = candidate_p_av[pid_candidate_full[i]]
        qlength = query_lengths[i]
        plength = passage_lengths[i]
        comb = np.hstack((query_embedding,p_embedding,qlength,plength))
        total_arr.append(comb)
        pid_list.append(pid_candidate_full[i])
    
    total_arr = np.array(total_arr)
    scores = model.predict(total_arr)
    
    for j in range(len(pid_list)):
        scores_dict.update({pid_list[j]: float(scores[j])})
        
    return scores_dict

def all_reranking(model, qid_list, df, query_embeddings, passage_embeddings, query_lengths, passage_lengths):
    re_ranked = {}
    for qid in qid_list:
        top_score = rank_candidate(candidate_df, qid, test_queries_av,
                              candidate_p_av, query_lengths, passage_lengths, top_scoring_model)
        re_ranked[qid] = top_score
    return re_ranked

lm_re_ranked = all_reranking(top_scoring_model, tes_qid, candidate_df, test_queries_av, candidate_p_av, query_lengths, passage_lengths)

np.save('lm_re_ranked',lm_re_ranked,allow_pickle=True)
lm_re_ranked = np.load('lm_re_ranked.npy',allow_pickle=True).item()

# Saving to txt file with top scoring 100 passages per query 
with open("LM.txt", "w") as f:
    for qid, pid_scores in lm_re_ranked.items():
        for rank, (pid, score) in enumerate(sorted(pid_scores.items(), key=lambda x: x[1], reverse=True)[:100]):
            algoname = "algoname2"  # Replace with your desired algorithm name
            line = f"<{qid} A2 {pid} {rank+1} {score} LM>\n"
            f.write(line)

