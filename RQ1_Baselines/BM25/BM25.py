### Used to calculate BM25 similarity between query and document first
import os
from tqdm import tqdm
import pandas as pd

DATA_DIR = '/home/kaixuan/patch_locating/PatchFinder/BM25'
DATA_TMP_DIR = '/home/kaixuan/patch_locating/PatchFinder/BM25'

### read bm25 similarity data
bm25_data = pd.read_csv(os.path.join(DATA_DIR, 'similarity_data_bm25okapi.csv'))
bm25_cve = bm25_data.groupby('cve')

### calculate the average rank of patch commit for each cve
### by using label to determine whether the commit is a patch commit
rank_info = pd.DataFrame(columns=['cve', 'rank'])
rank_info.to_csv(os.path.join(DATA_TMP_DIR, 'rankinfo_BM25.csv'), index=False)

for cve, group in tqdm(bm25_cve):
    # first sort the rows according to the similarity score
    sorted_group = group.sort_values(by='similarity', ascending=False)
    
    # create a mapping from original index to rank
    rank_mapping = {index: rank+1 for rank, (index, _) in enumerate(sorted_group.iterrows())}
    
    average_rank = 0
    # get rows corresponding to patch commits
    patch_rows = group[group['label'] == 1]
    ranks = []
    for _, row in patch_rows.iterrows():
        # get the rank using the rank mapping
        rank = rank_mapping[row.name]
        ranks.append(rank)
        
    try:
        average_rank = sum(ranks) / len(ranks)
    except ZeroDivisionError:
        print('cve: {} has no patch commit'.format(cve))
        average_rank = 1

    
    rank_info_iter = pd.DataFrame([[cve, average_rank]], columns=['cve', 'rank'])
    rank_info_iter.to_csv(os.path.join(DATA_TMP_DIR, 'rankinfo_BM25.csv'), mode='a', header=False, index=False)
    
### calculate the Top@K recall by using the rank info
k_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100]

def recall(k_list=k_list, save_path=os.path.join(DATA_TMP_DIR, 'recall_BM25.csv')):
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
def mrr(save_path=os.path.join(DATA_TMP_DIR, 'MRR_BM25.csv')):
    rank_info = pd.read_csv(os.path.join(DATA_TMP_DIR, 'rankinfo_BM25.csv'))
    rank_info['reciprocal_rank'] = rank_info['rank'].apply(lambda x: 1 / x)
    mrr = rank_info['reciprocal_rank'].sum() / len(rank_info)
    print('MRR: {}'.format(mrr))
    mrr_info = pd.DataFrame([[mrr]], columns=['MRR'])
    mrr_info.to_csv(save_path, index=False)
    print('MRR info saved')
    
### calculate the average manual efforts for Top@K
def manual_efforts(k_list=k_list, save_path=os.path.join(DATA_TMP_DIR, 'manualefforts_BM25.csv')):
    manual_efforts_info = pd.DataFrame(columns=['Top@k', 'manual_efforts'])
    manual_efforts_info.to_csv(save_path, index=False)
    
    for k in k_list:
        rank_info = pd.read_csv(os.path.join(DATA_TMP_DIR, 'rankinfo_BM25.csv'))
        rank_info['manual_efforts'] = rank_info['rank'].apply(lambda x: 1 if x <= k else k)
        manual_efforts = rank_info['manual_efforts'].sum() / len(rank_info)
        print('Top@{} manual efforts: {}'.format(k, manual_efforts))
        manual_efforts_info_iter = pd.DataFrame([[k, manual_efforts]], columns=['Top@k', 'manual_efforts'])
        manual_efforts_info_iter.to_csv(save_path, mode='a', header=False, index=False)

# def check_scores(file_path=os.path.join(DATA_DIR, 'similarity_data_bm25+1.6_0.3_0.7.csv')):
#     df = pd.read_csv(file_path)
#     negative_rows = df[df['similarity'] < 0]
#     print(negative_rows)
# check_scores()


recall()
mrr()
manual_efforts()

'''
Top@1 recall: 0.22175819586552517
Top@1 recall info saved
Top@5 recall: 0.3476717477552725
Top@5 recall info saved
Top@10 recall: 0.39674253497598666
Top@10 recall info saved
Top@20 recall: 0.4499895594069743
Top@20 recall info saved
Top@30 recall: 0.4821465859260806
Top@30 recall info saved
Top@50 recall: 0.5291292545416579
Top@50 recall info saved
Top@100 recall: 0.6007517226978493
Top@100 recall info saved
'''


'''
bm25 = BM25Okapi(corpus, k1=1.1, b=0.3)

Top@1 recall: 0.1252871163082063
Top@1 recall info saved
Top@5 recall: 0.21737314679473793
Top@5 recall info saved
Top@10 recall: 0.26477343913134266
Top@10 recall info saved
Top@20 recall: 0.31885571100438503
Top@20 recall info saved
Top@30 recall: 0.36061808310712046
Top@30 recall info saved
Top@50 recall: 0.4151179787011902
Top@50 recall info saved
Top@100 recall: 0.49885153476717475
Top@100 recall info saved
'''