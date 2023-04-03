# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 13:50:25 2023

@author: elfre
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F

# GETTING INPUT FEATURES

merged_df = pd.read_csv('saved_df.csv',header=0)
relevance_scores = list(merged_df.relevancy)
relevance_scores_tensor = torch.tensor(relevance_scores)

def str_to_array(num_str):
    num_str = num_str.strip('[]').split(',')
    num_array = np.array([float(x) for x in num_str])
    return num_array

# Apply the function to every row of the column
merged_df['passage embedding'] = merged_df['passage embedding'].apply(str_to_array)
merged_df['query embedding'] = merged_df['query embedding'].apply(str_to_array)

# make sure merged_val_df is in original state with no changes
query_embeddings = list(merged_df['query embedding'].values)
passage_embeddings = list(merged_df['passage embedding'].values)

# Getting query and passage lengths for training data
train = pd.read_csv("train_data.tsv",header=0,sep='\t',low_memory=False)

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

training_df['query length'] = training_df['queries'].apply(len)
training_df['passage length'] = training_df['passage'].apply(len)  

query_lengths = training_df['query length'].values
passage_lengths = training_df['passage length'].values   

del training_df
del train
del negative_df
del relevant_df
del irrelevant_df  
del negative_sample_size  

# Creating tensors 
query_tensor = torch.tensor(query_embeddings)
passage_tensor = torch.tensor(passage_embeddings)
relevance_scores_tensor = torch.tensor(relevance_scores)
query_len_tensor = torch.tensor(query_lengths)
passage_len_tensor = torch.tensor(passage_lengths)  
#%%
# Defining RNN module
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, lengths):
        # sort inputs by length
        lengths, sort_idx = lengths.sort(dim=0, descending=True)
        x = x[sort_idx]
        
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True)
        
        # forward pass
        h0 = torch.zeros(self.num_layers, x.batch_sizes[0], self.hidden_size).to(x.data.device)
        out, _ = self.rnn(x, h0)
        
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        
        # reorder outputs
        _, unsort_idx = sort_idx.sort(dim=0)
        out = out[unsort_idx]
        
        # fully connected layer and sigmoid activation
        out = self.fc(out)
        out = torch.sigmoid(out)

        return out

dataset = TensorDataset(query_tensor, passage_tensor, relevance_scores_tensor)

batch_size = 32
input_size = query_tensor.shape[1] + passage_tensor.shape[1]
criterion = nn.BCELoss()

shuffle = True
# drop_last = True as batches weren't even
loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,drop_last=True)

# Checking sizes as one batch was 17 not 32, so set 'drop_last' to True
# Checking all are same shape 
for batch_idx, (query, passage, rel_score) in enumerate(loader):
    print(f"Query shape: {query.shape}")
    print(f"Passage shape: {passage.shape}")
    print(f"Rel_score shape: {rel_score.shape}")
#%%
# predicting relevance of validation data

df = pd.read_csv('saved_val_df.csv',header=0)
relevance_scores_val = list(df.relevancy)
relevance_scores_tensor_val = torch.tensor(relevance_scores_val)
del df

merged_val_df = pd.read_csv('saved_val_df.csv',header=0)
merged_val_df['passage embedding'] = merged_val_df['passage embedding'].apply(str_to_array)
merged_val_df['query embedding'] = merged_val_df['query embedding'].apply(str_to_array)

# make sure merged_val_df is in original state with no changes
query_embeddings1 = merged_val_df['query embedding'].values
passage_embeddings1 = merged_val_df['passage embedding'].values

validation_query_tensor = torch.tensor(list(query_embeddings1))
validation_passage_tensor = torch.tensor(list(passage_embeddings1))

pid_list_full = list(merged_val_df.pid.values)
qid_list_full = list(merged_val_df.qid.values)

qid_list = list(set(qid_list_full))
pid_list = list(set(pid_list_full))

del merged_val_df
del query_embeddings1
del passage_embeddings1

# Define a data loader for the validation data
validation_dataset = TensorDataset(validation_query_tensor, validation_passage_tensor)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,drop_last=True)

# Checking sizes as one batch was 31 not 32, so set 'drop_last' to True
for batch_idx, (query, passage) in enumerate(validation_loader):
    print(f"Query shape: {query.shape}")
    print(f"Passage shape: {passage.shape}")

#%%
df = pd.read_csv("validation_data.tsv",header=0,sep='\t',low_memory=False)
qid_full = df['qid'].tolist()
pid_full = df['pid'].tolist()

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

def average_prec(top_pids):
    AP = []
    # loop through this as not all qids in log_reg_score
    for qid in list(top_pids.keys()):
        # get relevant pid(s): {pid:position}
        rels = relev[qid]
        retrieved = 0
        prec = 0
        # loop through pid(s)
        for i in list(top_pids[qid].keys()):
            # if pid(s) retrieved by bm25, continue, if not then AP is 0?
            if i in list(rels.keys()):
                # no of rel docs retrieved / tot docs retrieved
                rank = list(top_pids[qid].keys()).index(i)+1
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
                # if this pid is retrieved by bm25, then we get relevance etc
                # if bm25 does not retrieve pid then ndcg is 0?
                if i in list(lm_score[qid].keys()):
                    # get rank of that pid (+1 as indexing starts at 0)
                    rank = list(lm_score[qid].keys()).index(i)+1
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

#%%
def trainmodel(num_epochs, model, loader,criterion,optimizer):
    best_loss = float('inf')
    patience = 3 
    early_stopping_counter = 0
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
    
        for batch_idx, (query, passage, rel_score) in enumerate(loader):
            test_input = torch.cat((query,passage),1).float().unsqueeze(1)
            outputs = model(test_input, torch.ones(batch_size))
            print(batch_idx)
            # setting labels to same size as test_inputs
            labels = rel_score.view(-1,1,1)
            loss = criterion(outputs,labels.float())
            print(loss)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # worked up to here
            
            # accuracy
            predicted = torch.round(outputs)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
            # update epoch loss
            epoch_loss += loss.item()
    
            if batch_idx % 100 == 0:
                print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, batch_idx+1, len(loader), loss.item()))
        
        # average loss and accuracy over epoch
        avg_loss = epoch_loss / len(loader)
        accuracy = correct / total
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Validation loss did not improve for {patience} consecutive epochs. Stopping early...")
                break
        return accuracy,avg_loss

# Hyperparam tuning - only shown for one example here
# Repeated for all combinations of parameters shown in report

num_epochs = 3
batch_size = 32

model = RNN(input_size, 128, 3)

# loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), 0.1)

# Train the model using these hyperparameters
accuracy, avg_loss = trainmodel(num_epochs, model, loader, criterion, optimizer)
  
model.eval()
# Make predictions on the validation data
predictions = {}
with torch.no_grad():
    for batch_idx, (query, passage) in enumerate(validation_loader):
        
        inputs = torch.cat((query, passage),1).float().unsqueeze(1)

        relevance_scores = model(inputs,torch.ones(batch_size))

        probabilities = torch.sigmoid(relevance_scores)

        batch_size = query.shape[0]
        
        for i in range(batch_size):
            qid = qid_list_full[batch_idx * batch_size + i]
            pid = pid_list_full[batch_idx * batch_size + i]
            score = probabilities[i].item()

            if qid not in predictions:
                predictions[qid] = {}
            predictions[qid][pid] = score
        

top_pids = {}
for qid, pid_scores in predictions.items():
    sorted_pid_scores = sorted(pid_scores.items(), key=lambda x: x[1], reverse=True)
    top_pids[qid] = {pid: score for pid, score in sorted_pid_scores[:100]}
    
model_ap, model_map = average_prec(top_pids)
model_ndcg, model_meanndcg = ndcg(top_pids)
print(model_map)
print(model_meanndcg)
    
#%%
# Training best model for 5 epochs

# hidden size 64, num layers 1, lr 0.01
best_model = RNN(input_size, 64, 1)

# Define the loss function and optimizer
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(best_model.parameters(), 0.01)

# 5 epochs rather than 3
best_model_acc, best_model_loss = trainmodel(5, best_model, loader, criterion, optimizer)

best_model.eval()
# Make predictions on the validation data
predictions = {}
with torch.no_grad():
    for batch_idx, (query, passage) in enumerate(validation_loader):
        
        inputs = torch.cat((query, passage),1).float().unsqueeze(1)

        relevance_scores = best_model(inputs,torch.ones(batch_size))

        probabilities = torch.sigmoid(relevance_scores)

        batch_size = query.shape[0]
        
        for i in range(batch_size):
            qid = qid_list_full[batch_idx * batch_size + i]
            pid = pid_list_full[batch_idx * batch_size + i]
            score = probabilities[i].item()

            if qid not in predictions:
                predictions[qid] = {}
            predictions[qid][pid] = score

final_top_pids = {}
for qid, pid_scores in predictions.items():
    sorted_pid_scores = sorted(pid_scores.items(), key=lambda x: x[1], reverse=True)
    final_top_pids[qid] = {pid: score for pid, score in sorted_pid_scores[:100]}
    
best_model_ap, best_model_map = average_prec(final_top_pids)
best_model_ndcg, best_model_meanndcg = ndcg(final_top_pids)

#%%

# Re-ranking candidate passages

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

def candidate_ranking(qid,model):
    predictions = {}
    batch_size = 32
    q_embedding1 = test_queries_av[qid]
    index_list = [i for i,k in enumerate(qid_candidate_full) if k == qid]
    
    p_embeddings = []
    pid_list = []
    for i in index_list:
        p_embeddings.append(candidate_p_av[pid_candidate_full[i]])
        pid_list.append(pid_candidate_full[i])
        
    passage_tensor = torch.tensor(p_embeddings)
    num_passages = passage_tensor.size(0)
    query_tensor = torch.tensor(q_embedding1)
    pid_tensor = torch.tensor(pid_list)
    
    query_tensor = query_tensor.repeat(num_passages, 1)
    candidate_dataset = TensorDataset(query_tensor, passage_tensor,pid_tensor)
    shuffle = True
    candidate_loader = DataLoader(candidate_dataset, batch_size=32, shuffle=shuffle,drop_last=True)
    
    """
    for batch_idx, (query, passage,pid) in enumerate(candidate_loader):
        print(f"Query shape: {query.shape}")
        print(f"Passage shape: {passage.shape}")
        print(f"Pid shape:{pid.shape}")
    """
    # using best performing model, set to eval model
    model.eval()
    # Make predictions on the validation data
    predictions = {}
    with torch.no_grad():
        for batch_idx, (query, passage, pid) in enumerate(candidate_loader):
            
            inputs = torch.cat((query, passage),1).float().unsqueeze(1)
    
            relevance_scores = model(inputs,torch.ones(batch_size))
    
            probabilities = torch.sigmoid(relevance_scores)
            
            batch_size = query.shape[0]
            
            for i in range(batch_size):
                qid = qid
                pid_i = pid[i].item()
                score = probabilities[i].item()
    
               # if qid not in predictions:
                  #  predictions[qid] = {}
                predictions[pid_i] = score
                
    return predictions


def all_reranking(qid_list):
    re_ranked = {}
    for qid in qid_list:
        top_score = candidate_ranking(qid,best_model)
        re_ranked[qid] = top_score
    return re_ranked                
    
all_nn_scores = all_reranking(tes_qid)

np.save('all_nn_scores',all_nn_scores,allow_pickle=True)

with open("NN.txt", "w") as f:
    for qid, pid_scores in all_nn_scores.items():
        for rank, (pid, score) in enumerate(sorted(pid_scores.items(), key=lambda x: x[1], reverse=True)[:100]):
            algoname = "algoname2"  # Replace with your desired algorithm name
            line = f"<{qid} A2 {pid} {rank+1} {score} NN>\n"
            f.write(line)