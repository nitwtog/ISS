import os
import pickle

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

print('loading pickle file...')
#读打分文件
external_score_file_path = r'external_score_ex8_test2.pickle'
with open(external_score_file_path,"rb") as f:
    score_dict = pickle.load(f)

print('handle score file...')
#处理得分文件
score_mtric = torch.tensor(score_dict['score_list']).squeeze().T
batch_index = score_dict['index']
length_batch = len(score_dict['index'])
while(len(batch_index[-1]) != len(batch_index[-2])):
    batch_index[-1].append(0)
#load external data
data_files = {}
data_files["train"] = r'acl/small_external.csv'
external_datasets = load_dataset('csv', data_files=data_files)

# for top_num in list(range(500,3000,500))+list(range(3000,6000,1000)):
for top_num in [550,2600,2700,2800,2900]:
    print(f'now k of topk is {top_num}')
    print('find max score index...')
    #找最大得分点
    top_k = top_num
    vals, indices = score_mtric.topk(k=top_k, dim=-1, largest=True, sorted=True)
    indices = list(set(indices.view(-1).numpy().tolist()))
    external_index_list = torch.tensor(score_dict['index'])[indices].view(-1).numpy().tolist()
    external_index_list.sort()
    print('数据大小为：', len(external_index_list))

    subset = []
    text = external_datasets['train']['text']
    for i in tqdm(external_index_list):
        subset.append(text[i])

    #保存模型
    data_list=[]
    for i,text in enumerate(subset):
        data_list.append([i, text])
    df1= pd.DataFrame(data=data_list, columns=['id', 'text'])
    df1.to_csv(f'sample_external-byscore-{len(external_index_list)}-top{top_k}.csv',index=False)


