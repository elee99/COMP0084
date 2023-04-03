# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:49:42 2023

@author: elfre
"""

#%%
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
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence
import random
from time import strftime, gmtime
#%%
""" Pre-processing and Sampling
"""
train = pd.read_csv("train_data.tsv",header=0,sep='\t',low_memory=False)
#%%
stop_words = stopwords.words('english')

pattern2 = r"(?:[A-Z]\.)+|\b\w+(?:[-']\w+)*(?:-\d+\w+)*\b|http://[a-zA-Z0-9./]+|(?:[A-Za-z]{1,2}\.)+|[\w\']+|\.\.\.|\?\!|[.?!]|\$?\d+(?:\.\d+)?%?"  

def query_processing(file):
    """
    
    Parameters
    ----------
    file : Input dataframe.

    Returns
    -------
    passages: Dataframe with processed queries column.

    """
    # Read in file
    passages = file
    # Use custom pattern to tokenise as before for each passage
    passages['queries'] = [nltk.regexp_tokenize(passages['queries'].iloc[i],r'\s+', gaps=True) for i in range(len(passages['queries']))]
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

def passage_processing(file):
    """
    
    Parameters
    ----------
    file : Input dataframe.

    Returns
    -------
    passages: Dataframe with processed passages column.

    """
    # Read in file
    passages = file
    # Use custom pattern to tokenise as before for each passage
    passages['passage'] = [nltk.regexp_tokenize(passages['passage'].iloc[i],r'\s+', gaps=True) for i in range(len(passages['passage']))]
    # Lowercase all words 
    passages['passage'] = passages['passage'].apply(lambda sent: [w.lower() for w in sent])
    # Lemmatizing 
    passages['passage'] = passages['passage'].apply(lambda sent: [lemmatizer.lemmatize(word) for word in sent])
    # Removing unicode characters
    passages['passage'] = passages['passage'].apply(lambda sent: [unidecode(word) for word in sent])
    # Removing stop words
    passages['passage'] = passages['passage'].apply(lambda x: [item for item in x if item not in stop_words])
    # Removing punctuation and joining tokens back into a string
    passages['passage'] = passages['passage'].apply(lambda x: ''.join([c for c in x if c not in string.punctuation]))
    # Splitting the resulting string back into a list of words
    passages['passage'] = passages['passage'].apply(lambda x: x.split())
    
    return passages
#%%

"""Negative sampling"""

# Split passages into relevant and irrelevant based on 'relevancy' column
relevant_df = train[train['relevancy'] == 1]
irrelevant_df = train[train['relevancy'] == 0]

# Randomly sample irrelevant passages to balance the data
negative_sample_size = len(relevant_df)
# Taking irrelevant sample of 100 times the size of the relevant df
negative_df = irrelevant_df.sample(negative_sample_size*100, random_state=42)

# Combine relevant and negative passages into a single dataframe
training_df = pd.concat([relevant_df, negative_df])

# Shuffle training data
training_df = training_df.sample(frac=1, random_state=42).reset_index(drop=True)

""" Validation data """
validation_df = pd.read_csv("validation_data.tsv",header=0,sep='\t',low_memory=False)

queries_val = query_processing(validation_df)

passages_val = passage_processing(validation_df)

with open('val_passages.txt', 'w') as f:
    for passage in passages_val['passage']:
        f.write(' '.join(passage) + '\n')

sent_val_p = LineSentence('val_passages.txt')

val_passages_embed = Word2Vec(sent_val_p,sg=1,min_count=1)

with open('val_queries.txt', 'w') as f:
    for passage in queries_val['queries']:
        f.write(' '.join(passage) + '\n')

sent_val_q = LineSentence('val_queries.txt')

val_queries_embed = Word2Vec(sent_val_q,sg=1,min_count=1)

def average_embed(train_proc,qid,embed):
    """
    
    Parameters
    ----------
    train_proc: Dataframe with queries and passages.
    qid: Specify either pid or qid.
    embed: Query or Passage average embeddings.

    Returns
    -------
    av_embeddings: Averaged query or passage embeddings.
    """
    av_embeddings = {}
    for i in range(len(train_proc)):
        num = len(train_proc.iloc[i])
        if num != 0:
            embed_av = sum(embed.wv[train_proc.iloc[i]])/num
            
            av_embeddings.update({qid.iloc[i]:embed_av})
    return av_embeddings


val_queries_embed_av = average_embed(validation_df['queries'],validation_df['qid'],val_queries_embed)
val_passages_embed_av = average_embed(validation_df['passage'],validation_df['pid'],val_passages_embed)

# SAVING AVERAGE QUERY EMBEDDINGS FOR VAL DATA
val_queries_embed_av = np.save('val_q_av',val_queries_embed_av,allow_pickle=True)
val_passages_embed_av = np.save('val_p_av',val_passages_embed_av,allow_pickle=True)

# To get as dictionaries call .item()
val_queries_embed_av = np.load('val_q_av.npy',allow_pickle=True).item()
val_passages_embed_av = np.load('val_p_av.npy',allow_pickle=True).item()

# Merged df for validation data
val_q_df = pd.DataFrame.from_dict(val_queries_embed_av, orient='index')
val_q_df.index.name = 'qid'
val_q_df.reset_index(inplace=True)
to_list = lambda row: list(row.values)
val_q_df['query embedding'] = val_q_df.iloc[:, 1:].apply(to_list, axis=1)
val_q_df = val_q_df.drop(val_q_df.columns[1:101],axis=1)

val_p_df = pd.DataFrame.from_dict(val_passages_embed_av, orient='index')
val_p_df.index.name = 'pid'
val_p_df.reset_index(inplace=True)
to_list = lambda row: list(row.values)
val_p_df['passage embedding'] = val_p_df.iloc[:, 1:].apply(to_list, axis=1)
val_p_df = val_p_df.drop(val_p_df.columns[1:101],axis=1)

merged_val_df = pd.merge(validation_df, val_q_df, on='qid', how='left')
merged_val_df = pd.merge(merged_val_df, val_p_df, on='pid', how='left')
merged_val_df = merged_val_df.drop(columns=['queries','passage'])
merged_val_df.head()

# SAVING
saved_val_df = merged_val_df.to_csv('saved_val_df.csv',index=False)
merged_val_df = pd.read_csv('saved_val_df.csv',header=0)

# Getting X test and y test
p_vals = np.vstack(merged_val_df['passage embedding'].values)
q_val = np.vstack(merged_val_df['query embedding'].values)
X_test = np.hstack((p_vals,q_val))
y_test = validation_df['relevancy'].values

# SAVING 
X_test = np.save('X_test',X_test,allow_pickle=True)
y_test = np.save('y_test',y_test,allow_pickle=True)
X_test = np.load('X_test.npy',allow_pickle=True)
y_test = np.load('y_test.npy',allow_pickle=True)

# Getting query and passage length to be used in next task
validation_df['query length'] = validation_df['queries'].apply(len)
validation_df['passage length'] = validation_df['passage'].apply(len)

merged_df2 = pd.merge(validation_df, val_q_df, on='qid', how='left')
merged_df2 = pd.merge(merged_df2, val_p_df, on='pid', how='left')
merged_df2 = merged_df2.drop(columns=['queries','passage'])
merged_df2.head()
merged_df2.columns
# wasn't saving
merged_df2.to_csv('val_df_length.csv',index=False)

q_length = np.vstack(merged_df2['query length'].values)
p_length = np.vstack(merged_df2['passage length'].values)
X_test_2 = np.hstack((q_length,p_length,q_val,p_vals))

X_test_2 = np.save('X_test_2',X_test_2,allow_pickle=True)

#%%

"""Training Data"""

passages = passage_processing(training_df)
queries = query_processing(training_df)

""" Passage Embeddings:
"""
with open('passage_embed.txt', 'w') as f:
    for passage in training_df['passage']:
        f.write(' '.join(passage) + '\n')

sent = LineSentence('passage_embed.txt')

passage_embed = Word2Vec(sent,sg=1,min_count=1)

""" Checking embeddings contain vocabulary words
"""
def check_vocab(query, model):
    
    for word in query:
        if word not in model.wv.key_to_index:
            return False
    return True

training_df['in_vocab'] = training_df['queries'].apply(lambda x: check_vocab(x, query_embed))
training_df[training_df['in_vocab']==False]

""" Query Embeddings
"""
with open('query_embed.txt', 'w') as f:
    for passage in training_df['queries']:
        f.write(' '.join(passage) + '\n')

sent_query = LineSentence('query_embed.txt')

query_embed = Word2Vec(sent_query,sg=1,min_count=1)

""" Averaging Embeddings
"""

passage_av = average_embed(training_df['passage'],training_df['pid'],passage_embed)
query_av = average_embed(training_df['queries'],training_df['qid'],query_embed)

# SAVING AVERAGE EMBEDDINGS
passage_av = np.save('passage_av',passage_av,allow_pickle=True)
query_av = np.save('query_av',query_av,allow_pickle=True)

passage_av = np.load('passage_av.npy',allow_pickle=True).item()
query_av = np.load('query_av.npy',allow_pickle=True).item()

#%%
""" Converting data into dataframe with embedddings
"""

query_emb_df = pd.DataFrame.from_dict(query_av, orient='index')
query_emb_df.index.name = 'qid'
query_emb_df.reset_index(inplace=True)
to_list = lambda row: list(row.values)
query_emb_df['query embedding'] = query_emb_df.iloc[:, 1:].apply(to_list, axis=1)
query_emb_df = query_emb_df.drop(query_emb_df.columns[1:101],axis=1)

passage_emb_df = pd.DataFrame.from_dict(passage_av, orient='index')
passage_emb_df.index.name = 'pid'
passage_emb_df.reset_index(inplace=True)
to_list = lambda row: list(row.values)
passage_emb_df['passage embedding'] = passage_emb_df.iloc[:, 1:].apply(to_list, axis=1)
passage_emb_df = passage_emb_df.drop(passage_emb_df.columns[1:101],axis=1)

merged_df = pd.merge(training_df, query_emb_df, on='qid', how='left')
merged_df = pd.merge(merged_df, passage_emb_df, on='pid', how='left')
merged_df = merged_df.drop(columns=['queries','passage'])
merged_df.head()

# SAVING DF
saved_df = merged_df.to_csv('saved_df.csv',index=False)

""" Creating X_train and y_train arrays
"""
test = np.vstack(merged_df['passage embedding'].values)
test2 = np.vstack(merged_df['query embedding'].values)
X_train = np.hstack((test,test2))
y_train = training_df['relevancy'].values

# SAVING 
X_train = np.save('X_train',X_train,allow_pickle=True)
y_train = np.save('y_train',y_train,allow_pickle=True)
X_train = np.load('X_train.npy',allow_pickle=True)
y_train = np.load('y_train.npy',allow_pickle=True)

# Adding query and passage length as features to be used in next task 
# Will have same y_test / y_train

# Train 
training_df['query length'] = training_df['queries'].apply(len)
training_df['passage length'] = training_df['passage'].apply(len)

merged_train = pd.merge(training_df,query_emb_df,on='qid',how='left')
merged_train = pd.merge(merged_train,passage_emb_df,on='pid',how='left')
merged_train = merged_train.drop(columns=['queries','passage'])

len_q_tr = np.vstack(merged_train['query length'].values)
len_p_tr = np.vstack(merged_train['passage length'].values)
q_values_tr = np.vstack(merged_train['query embedding'].values)
p_values_tr = np.vstack(merged_train['passage embedding'].values)
X_train_2 = np.hstack((len_q_tr,len_p_tr,q_values_tr,p_values_tr))

X_train_2 = np.save('X_train_2',X_train_2,allow_pickle=True)

#%%
""" Logistic Regression Functions

 - Weight initialisation
 - Sigmoid function
 - Loss function
 - Gradient Descent
 - Prediction
 
"""

def weight_init(n_features):
    """
    
    Parameters
    ----------
    n_features : Number of features (200).

    Returns
    -------
    w : Weights.
    b : Bias.

    """
    w = np.zeros((1,n_features))
    b = 0
    return w,b

def sigmoid(res):
    """

    Parameters
    ----------
    res : Takes in dot product of X and w, plus b.

    Returns
    -------
    final_result : Computes sigmoid on Xw + b.

    """
    final_res = 1/(1+np.exp(-res))
    return final_res

def grad_func(w, b, X, Y):
    """

    Parameters
    ----------
    w : Weights.
    b : Bias.
    X : Input X - Query and Passage embeddings.
    Y : Input labels - Relevancy.

    Returns
    -------
    grads : Gradients of w and b.
    loss : Loss function.

    """
    m = X.shape[0]
    
    final_res = sigmoid(np.dot(w,X.T)+b)
    Y_T = Y.T
    loss = (-1/m)*(np.sum((Y_T*np.log(final_res)) + ((1-Y_T)*(np.log(1-final_res)))))
    
    dw = (1/m)*(np.dot(X.T, (final_res-Y.T).T))
    db = (1/m)*(np.sum(final_res-Y.T))
    
    grads = {"dw": dw, "db": db}
    
    return grads, loss

def grad_descent(w, b, X, Y, lr, max_iter, tol=1e-4):
    """

    Parameters
    ----------
    w : Weights.
    b : Bias.
    X : Input X (Query and Passage Embeddings).
    Y : Input labels (Relevancy).
    lr : Learning rate.
    max_iter : Maximum number of iterations.

    Returns
    -------
    coeff : Final weights and biases.
    grad : Final gradients.
    losses : Losses.

    """
    losses = []
    prev_loss = float('inf')
    
    for i in range(max_iter):
        grads, loss = grad_func(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        w = w - lr * dw.T
        b = b - lr * db
       
        losses.append(loss)
        
        if abs(loss - prev_loss) < tol:
            break
        
        prev_loss = loss

    coeff = {"w": w, "b": b}
    grad = {"dw": dw, "db": db}
    return coeff, grad, losses


def pred(final_pred, m):
    """
    
    Parameters
    ----------
    final_pred: Result of sigmoid function on test X array, w and b.
    m: Shape of train or test X.
    
    Returns
    -------
    y_pred: Predictions.
    
    """
    y_pred = np.zeros((1,m))
    for i in range(final_pred.shape[1]):
        # if prob >0.5, classify as relevant
        if final_pred[0][i] > 0.5:
            y_pred[0][i] = 1
    return y_pred

""" Logistic Regression experiment:
"""

n_features = X_train.shape[1]
w, b = weight_init(n_features)

lr_rates = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]

coeff, grad, losses = grad_descent(w, b, X_train, y_train, lr=0.01,max_iter=500)
w = coeff["w"]
b = coeff["b"]

# saving weights and bias
np.save('log_reg_weights',w,allow_pickle=True)
np.save('log_reg_bias',b,allow_pickle=True)
w = np.load('log_reg_weights.npy',allow_pickle=True)
b = np.load('log_reg_bias.npy',allow_pickle=True)

# Training loss for different learning rates

loss_list = [[] for _ in range(len(lr_rates))]

# Run gradient descent for each learning rate and record the training loss
print (strftime("%Y-%m-%d %H:%M:%S", gmtime()))

for i, learning_rate in enumerate(lr_rates):
    coeff, grad, losses = grad_descent(w, b, X_train, y_train, lr_rates[i], max_iter=500)
    loss_list[i] = losses
    
np.save('loss_list_500',loss_list,allow_pickle=True)
loss_list2 = list(np.load('loss_list_500.npy',allow_pickle=True))
del loss_list2

plt.rcParams['font.family'] = 'serif'
# Plot the learning curves for each learning rate
plt.figure()
for i, learning_rate in enumerate(lr_rates):
    plt.plot(loss_list[i], label=f"lr={learning_rate}")
plt.xlabel("Number of iterations")
plt.ylabel("Training loss")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('training_loss.pdf',bbox_inches='tight')
plt.show()

#%%

# for prediction, need to go through each qid in val data, get corresponding pids
# predict that pids relevancy
# sort in decreasing order based on probability score
pid_list_full = list(merged_val_df.pid.values)
qid_list_full = list(merged_val_df.qid.values)

qid_list = list(set(qid_list_full))
pid_list = list(set(pid_list_full))

# probs easier to do this first than calling it every time in predict_pid

def str_to_array(num_str):
    num_str = num_str.strip('[]').split(',')
    num_array = np.array([float(x) for x in num_str])
    return num_array

# Apply the function to every row of the column
merged_val_df['passage embedding'] = merged_val_df['passage embedding'].apply(str_to_array)
merged_val_df['query embedding'] = merged_val_df['query embedding'].apply(str_to_array)

# make sure merged_val_df is in original state with no changes
query_embeddings1 = merged_val_df['query embedding'].values
passage_embeddings1 = merged_val_df['passage embedding'].values

def predict_pid(df, qid, query_embeddings, passage_embeddings):
    index_q = qid_list_full.index(qid)
    query_embedding = query_embeddings[index_q]
    
    ind_list = np.where(qid_list_full == qid)[0]
    p_embeddings = passage_embeddings1[ind_list]
    
    scores_dict = {}
    for i, embedding in enumerate(p_embeddings):
        comb = np.hstack((query_embedding, embedding))
        score = sigmoid(np.dot(w, comb.T) + b)
        pid = pid_list_full[ind_list[i]]
        scores_dict.update({pid:float(score)})
    
    top = dict(sorted(scores_dict.items(),key=lambda x:x[1],reverse=True)[:100])
    
    return top

log_reg_score = {}
for qid in range(len(qid_list)):
    top_score = predict_pid(merged_val_df, qid_list[qid], query_embeddings1, passage_embeddings1)
    if len(top_score) == 100:
        log_reg_score[qid_list[qid]] = top_score
# len 1123, must mean some didnt have 100 corresponding pids..?

# using lr=0.01 and 500 iter for the weights and bias
np.save('lr_scores',log_reg_score,allow_pickle=True)
lr_scores = np.load('lr_scores.npy',allow_pickle=True).item()

#%%
df = pd.read_csv("validation_data.tsv",header=0,sep='\t',low_memory=False)
qid_full = df['qid'].tolist()
pid_full = df['pid'].tolist()

def get_relevant(df):
    relev = {}
    not_relev = {}
    relevancy = df['relevancy']
    qid_pid = zip(qid_full, pid_full)
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

def average_prec():
    AP = []
    # loop through this as not all qids in log_reg_score
    for qid in list(log_reg_score.keys()):
        # get relevant pid(s): {pid:position}
        rels = relev[qid]
        retrieved = 0
        prec = 0
        # loop through pid(s)
        for i in list(log_reg_score[qid].keys()):
            # if pid(s) retrieved by bm25, continue, if not then AP is 0?
            if i in list(rels.keys()):
                # no of rel docs retrieved / tot docs retrieved
                rank = list(log_reg_score[qid].keys()).index(i)+1
                # total docs retrieved is same as rank of current i 
                retrieved += 1
                # need to divide by total relevant docs?
                prec += (retrieved/rank)/len(rels.keys())
        
            else:
                prec += 0
        AP.append(prec) 
    return AP, np.mean(AP)

AP, mean_AP = average_prec() # 0.00556

def ndcg(lr_score):
    NDCG = []
    for qid in lr_score.keys():
        rels = relev[qid]  # relevant passages
        # i is pid(s)
        # loop through relevant pid(s)
        none_found = [i not in lr_score[qid].keys() for i in rels.keys()]
        if none_found == True:
            ndcg_ = 0  
        else:
            opt = {}
            for i in list(rels.keys()):
                # set dcg to 0
                dg = 0
                # if this pid is retrieved by bm25, then we get relevance etc
                # if bm25 does not retrieve pid then ndcg is 0?
                if i in list(lr_score[qid].keys()):
                    # get rank of that pid (+1 as indexing starts at 0)
                    rank = list(lr_score[qid].keys()).index(i)+1
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

NDCG, meanNDCG = ndcg(lr_scores) # 0.0232
#%%

"""Re-ranking candidate passages"""

# Pre-processing
candidate = pd.read_csv("candidate-passages-top1000.tsv",sep='\t',names=['qid','pid','queries','passage'])
test_query = pd.read_csv("test-queries.tsv",sep='\t',names=['qid','queries'])

def candidate_passage_processing(file):
    """
    
    Parameters
    ----------
    file : Input dataframe.

    Returns
    -------
    passages: Dataframe with processed passages column.

    """
    # Read in file
    passages = file
    # Use custom pattern to tokenise as before for each passage
    passages['passage'] = [nltk.regexp_tokenize(passages['passage'].iloc[i],r'\s+', gaps=True) for i in range(len(passages['passage']))]
    # Lowercase all words 
    passages['passage'] = passages['passage'].apply(lambda sent: [w.lower() for w in sent])
    # Lemmatizing 
    passages['passage'] = passages['passage'].apply(lambda sent: [lemmatizer.lemmatize(word) for word in sent])
    # Removing unicode characters
    passages['passage'] = passages['passage'].apply(lambda sent: [unidecode(word) for word in sent])
    # Removing stop words
    passages['passage'] = passages['passage'].apply(lambda x: [item for item in x if item not in stop_words])
    # Removing punctuation and joining tokens back into a string
    passages['passage'] = passages['passage'].apply(lambda x: ''.join([c for c in x if c not in string.punctuation]))
    # Splitting the resulting string back into a list of words
    passages['passage'] = passages['passage'].apply(lambda x: x.split())
    
    return passages

def candidate_query_processing(file):
    """
    
    Parameters
    ----------
    file : Input dataframe.

    Returns
    -------
    passages: Dataframe with processed queries column.

    """
    # Read in file
    passages = file
    # Use custom pattern to tokenise as before for each passage
    passages['queries'] = [nltk.regexp_tokenize(passages['queries'].iloc[i],r'\s+', gaps=True) for i in range(len(passages['queries']))]
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

candidate_passages = candidate_passage_processing(candidate)
test_queries = query_processing(test_query)

"""Embeddings"""

with open('candidate_passage_embed.txt', 'w') as f:
    for passage in candidate_passages['passage']:
        f.write(' '.join(passage) + '\n')

sent = LineSentence('candidate_passage_embed.txt')

candidate_passage_embed = Word2Vec(sent,sg=1,min_count=1)

with open('test_query_embed.txt', 'w') as f:
    for passage in test_queries['queries']:
        f.write(' '.join(passage) + '\n')

sent_query = LineSentence('test_query_embed.txt')

test_query_embed = Word2Vec(sent_query,sg=1,min_count=1)

def average_embed(train_proc,qid,embed):
    """
    
    Parameters
    ----------
    train_proc: Dataframe with queries and passages.
    qid: Specify either pid or qid.
    embed: Query or Passage embeddings.

    Returns
    -------
    av_embeddings: Averaged query or passage embeddings.
    """
    av_embeddings = {}
    for i in range(len(train_proc)):
        num = len(train_proc.iloc[i])
        if num != 0:
            embed_av = sum(embed.wv[train_proc.iloc[i]])/num
            
            av_embeddings.update({qid.iloc[i]:embed_av})
    return av_embeddings

test_queries_embed_av = average_embed(test_queries['queries'],test_queries['qid'],test_query_embed)

candidate_embed_av = average_embed(candidate_passages['passage'],candidate_passages['pid'],candidate_passage_embed)

# SAVING
np.save('test_queries_av',test_queries_embed_av,allow_pickle=True)
np.save('candidate_p_av',candidate_embed_av,allow_pickle=True)

test_queries_av = np.load('test_queries_av.npy',allow_pickle=True).item()
candidate_p_av = np.load('candidate_p_av.npy',allow_pickle=True).item()

"""New dataframe"""

query_emb_df = pd.DataFrame.from_dict(test_queries_embed_av, orient='index')
query_emb_df.index.name = 'qid'
query_emb_df.reset_index(inplace=True)
to_list = lambda row: list(row.values)
query_emb_df['query embedding'] = query_emb_df.iloc[:, 1:].apply(to_list, axis=1)
query_emb_df = query_emb_df.drop(query_emb_df.columns[1:101],axis=1)

passage_emb_df = pd.DataFrame.from_dict(candidate_embed_av, orient='index')
passage_emb_df.index.name = 'pid'
passage_emb_df.reset_index(inplace=True)
to_list = lambda row: list(row.values)
passage_emb_df['passage embedding'] = passage_emb_df.iloc[:, 1:].apply(to_list, axis=1)
passage_emb_df = passage_emb_df.drop(passage_emb_df.columns[1:101],axis=1)

merged_df = pd.merge(candidate_passages, query_emb_df, on='qid', how='left')
merged_df = pd.merge(merged_df, passage_emb_df, on='pid', how='left')
merged_df = merged_df.drop(columns=['queries','passage'])
merged_df.head()
merged_df.columns

# SAVING DF
candidate_df = merged_df.to_csv('candidate_df.csv',index=False)

del query_emb_df
del passage_emb_df

candidate_df = pd.read_csv('candidate_df.csv',header=0)
#%%
def candidate_passage_processing2(file):
    """
    
    Parameters
    ----------
    file : Input dataframe.

    Returns
    -------
    passages: Dataframe with processed passages column.

    """
    # Read in file
    passages = file
    # Use custom pattern to tokenise as before for each passage
    passages['passage'] = [nltk.regexp_tokenize(passages['passage'].iloc[i],r'\s+', gaps=True) for i in range(len(passages['passage']))]
    # Lowercase all words 
    passages['passage'] = passages['passage'].apply(lambda sent: [w.lower() for w in sent])
    # Lemmatizing 
    passages['passage'] = passages['passage'].apply(lambda sent: [lemmatizer.lemmatize(word) for word in sent])
    # Removing unicode characters
    passages['passage'] = passages['passage'].apply(lambda sent: [unidecode(word) for word in sent])
    # Removing stop words
    passages['passage'] = passages['passage'].apply(lambda x: [item for item in x if item not in stop_words])
    # Removing punctuation and joining tokens back into a string
    passages['queries'] = passages['queries'].apply(lambda x: [item for item in x if item not in string.punctuation])
    
    return passages

candidate_passages = candidate_passage_processing2(candidate)
candidate_passages.head()

def candidate_query_processing2(file):
    """
    
    Parameters
    ----------
    file : Input dataframe.

    Returns
    -------
    passages: Dataframe with processed queries column.

    """
    # Read in file
    passages = file
    # Use custom pattern to tokenise as before for each passage
    passages['queries'] = [nltk.regexp_tokenize(passages['queries'].iloc[i],r'\s+', gaps=True) for i in range(len(passages['queries']))]
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

test_queries = candidate_query_processing2(test_query)

qid_candidate_full = list(candidate_passages.qid)
pid_candidate_full = list(candidate_passages.pid)
qid_candidate_list = list(set(qid_candidate_full))
pid_candidate_list = list(set(pid_candidate_full))

tes_qid = list(test_queries.qid)
np.save('test_qids',tes_qid,allow_pickle=True)

# For later use
candidate_passages['passage length'] = candidate_passages['passage'].apply(len)
candidate_passages['query length'] = candidate_passages['queries'].apply(len)

query_lengths = candidate_passages['query length'].values
passage_lengths = candidate_passages['passage length'].values

# SAVING
np.save('candidate_qids',qid_candidate_full,allow_pickle=True)
np.save('candidate_pids',pid_candidate_full,allow_pickle=True)
np.save('candidate_qlength',query_lengths,allow_pickle=True)
np.save('candidate_plength',passage_lengths,allow_pickle=True)

candidate_qids_full = np.load('candidate_qids.npy',allow_pickle=True)
candidate_pids_full = np.load('candidate_pids.npy',allow_pickle=True) 
qlengths = np.load('candidate_qlength.npy',allow_pickle=True)
plengths = np.load('candidate_plength.npy',allow_pickle=True)

qid_candidate_list = list(set(candidate_qids_full))
pid_candidate_list = list(set(candidate_pids_full))

def rank_candidate(qid, query_embeddings, passage_embeddings):
    index_list = [i for i,k in enumerate(candidate_qids_full) if k == qid]
    # is this right? dict form not list
    query_embedding = test_queries_av[qid]
    
    scores_dict = {}
    
    for i in index_list:
        p_embedding = candidate_p_av[candidate_pids_full[i]]
        comb = np.hstack((query_embedding,p_embedding))
        score = sigmoid(np.dot(w,comb.T)+b)
        pid = candidate_pids_full[i]
        scores_dict.update({pid:float(score)})

    return scores_dict

def all_reranking(qid_list, query_embeddings, passage_embeddings):
    re_ranked = {}
    for qid in qid_list:
        top_score = rank_candidate(qid, test_queries_av,
                              candidate_p_av)
        re_ranked[qid] = top_score
    return re_ranked

lr_re_ranked = all_reranking(tes_qid, test_queries_av, candidate_p_av)
np.save('final_lr_ranking',lr_re_ranked,allow_pickle=True)
lr_re_ranked = np.load('final_lr_ranking.npy',allow_pickle=True).item()

with open("LR.txt", "w") as f:
    for qid, pid_scores in lr_re_ranked.items():
        for rank, (pid, score) in enumerate(sorted(pid_scores.items(), key=lambda x: x[1], reverse=True)[:100]):
            algoname = "algoname2"  # Replace with your desired algorithm name
            line = f"<{qid} A2 {pid} {rank+1} {score} LR>\n"
            f.write(line)
#%%