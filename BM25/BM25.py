### Used to calculate BM25 similarity between query and document first
import os
from tqdm import tqdm
import pandas as pd

DATA_DIR = '/data/kaixuan/ramdisk/data'
DATA_TMP_DIR = '/data/kaixuan/data_tmp'

### read bm25 similarity data
bm25_data = pd.read_csv(os.path.join(DATA_DIR, 'similarity_data_bm25.csv'))
bm25_cve = bm25_data.groupby('cve')

### calculate the average rank of patch commit for each cve
### by using label to determine whether the commit is a patch commit
rank_info = pd.DataFrame(columns=['cve', 'rank'])
rank_info.to_csv(os.path.join(DATA_TMP_DIR, 'rankinfo_BM25.csv'), index=False)

for cve, group in tqdm(bm25_cve):
    # first sort the rows according to the similarity score
    group = group.sort_values(by='similarity', ascending=False)
    average_rank = 0
    #### maybe there are multiple patch commits for one cve
    patch_rows = group[group['label'] == 1]
    ranks = []
    for _, row in patch_rows.iterrows():
        ### get the rank by using the index of the row
        rank = group.index.get_loc(row.name) + 1
        ranks.append(rank)
    average_rank = sum(ranks) / len(ranks)
    rank_info_iter = pd.DataFrame([[cve, average_rank]], columns=['cve', 'rank'])
    rank_info_iter.to_csv(os.path.join(DATA_TMP_DIR, 'rankinfo_BM25.csv'), mode='a', header=False, index=False)
    
### calculate the Top@K recall by using the rank info
def recall(k_list=[1, 5, 10, 20, 30, 50, 100], save_path=os.path.join(DATA_TMP_DIR, 'recall_BM25.csv')):
    recall_info = pd.DataFrame(columns=['Top@k', 'recall'])
    recall_info.to_csv(save_path, index=False)

    for k in k_list:
        rank_info = pd.read_csv(os.path.join(DATA_TMP_DIR, 'rankinfo_BM25.csv'))
        rank_info['recall'] = rank_info['rank'].apply(lambda x: 1 if x <= k else 0)
        recall = rank_info['recall'].sum() / len(rank_info)
        print('Top@{} recall: {}'.format(k, recall))
        recall_info_iter = pd.DataFrame([[k, recall]], columns=['Top@k', 'recall'])
        recall_info_iter.to_csv(save_path, mode='a', header=False, index=False)
        print('Top@{} recall info saved'.format(k))

### calculate the MRR
def mrr(rank_info, save_path=os.path.join(DATA_TMP_DIR, 'MRR_BM25.csv')):
    rank_info = pd.read_csv(os.path.join(DATA_TMP_DIR, 'rankinfo_BM25.csv'))
    rank_info['reciprocal_rank'] = rank_info['rank'].apply(lambda x: 1 / x)
    mrr = rank_info['reciprocal_rank'].sum() / len(rank_info)
    print('MRR: {}'.format(mrr))
    mrr_info = pd.DataFrame([[mrr]], columns=['MRR'])
    mrr_info.to_csv(save_path, index=False)
    print('MRR info saved')
    
### calculate the average manual efforts for Top@K
def manual_efforts(k_list=[1, 5, 10, 20, 30, 50, 100], save_path=os.path.join(DATA_TMP_DIR, 'manualefforts_BM25.csv')):
    manual_efforts_info = pd.DataFrame(columns=['Top@k', 'manual_efforts'])
    manual_efforts_info.to_csv(save_path, index=False)
    
    for k in k_list:
        rank_info = pd.read_csv(os.path.join(DATA_TMP_DIR, 'rankinfo_BM25.csv'))
        rank_info['manual_efforts'] = rank_info['rank'].apply(lambda x: 1 if x <= k else 0)
        manual_efforts = rank_info['manual_efforts'].sum() / len(rank_info)
        print('Top@{} manual efforts: {}'.format(k, manual_efforts))
        manual_efforts_info_iter = pd.DataFrame([[k, manual_efforts]], columns=['Top@k', 'manual_efforts'])
        manual_efforts_info_iter.to_csv(save_path, mode='a', header=False, index=False)

recall()
