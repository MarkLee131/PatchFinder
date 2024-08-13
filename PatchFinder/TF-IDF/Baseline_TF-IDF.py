### Used to calculate TF-IDF similarity between query and document first
# TF-IDF similarity
import os
from tqdm import tqdm
import pandas as pd

DATA_DIR = '/mnt/local/Baselines_Bugs/PatchFinder/TF-IDF'
DATA_TMP_DIR = '/mnt/local/Baselines_Bugs/PatchFinder/TF-IDF/tmp_0830'
os.makedirs(DATA_TMP_DIR, exist_ok=True)
    
### calculate the Top@K recall by using the rank info
def recall(k_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100], save_path=os.path.join(DATA_TMP_DIR, 'recall_TFIDF.csv')):
    recall_info = pd.DataFrame(columns=['Top@k', 'recall'])
    recall_info.to_csv(save_path, index=False)

    for k in k_list:
        rank_info = pd.read_csv(os.path.join(DATA_TMP_DIR, 'rankinfo_TDIDF.csv'))
        rank_info['recall'] = rank_info['rank'].apply(lambda x: 1 if x <= k else 0)
        recall = rank_info['recall'].sum() / len(rank_info)
        print('Top@{} recall: {}'.format(k, recall))
        recall_info_iter = pd.DataFrame([[k, recall]], columns=['Top@k', 'recall'])
        recall_info_iter.to_csv(save_path, mode='a', header=False, index=False)
        print('Top@{} recall info saved'.format(k))

### calculate the MRR
def mrr(save_path=os.path.join(DATA_TMP_DIR, 'MRR_TFIDF.csv')):
    rank_info = pd.read_csv(os.path.join(DATA_TMP_DIR, 'rankinfo_TDIDF.csv'))
    rank_info['reciprocal_rank'] = rank_info['rank'].apply(lambda x: 1 / x)
    mrr = rank_info['reciprocal_rank'].sum() / len(rank_info)
    print('MRR: {}'.format(mrr))
    mrr_info = pd.DataFrame([[mrr]], columns=['MRR'])
    mrr_info.to_csv(save_path, index=False)
    print('MRR info saved')
    
### calculate the average manual efforts for Top@K
'''
if rank <= k: 
    manual_efforts = rank
else: 
    manual_efforts = k
'''
def manual_efforts(k_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100], save_path=os.path.join(DATA_TMP_DIR, 'manualefforts_TFIDF.csv')):
    manual_efforts_info = pd.DataFrame(columns=['Top@k', 'manual_efforts'])
    manual_efforts_info.to_csv(save_path, index=False)
    
    for k in tqdm(k_list):
        rank_info = pd.read_csv(os.path.join(DATA_TMP_DIR, 'rankinfo_TDIDF.csv'))
        rank_info['manual_efforts'] = rank_info['rank'].apply(lambda x: x if x <= k else k)
        manual_efforts = rank_info['manual_efforts'].sum() / len(rank_info)
        print('Top@{} manual efforts: {}'.format(k, manual_efforts))
        manual_efforts_info_iter = pd.DataFrame([[k, manual_efforts]], columns=['Top@k', 'manual_efforts'])
        manual_efforts_info_iter.to_csv(save_path, mode='a', header=False, index=False)



if __name__ == "__main__":
    
    
    # ### step 0: preprocess the test data, just once, then use the saved csv file.
    # tf_idf_rawdata = pd.read_csv('/mnt/local/Baselines_Bugs/PatchFinder/TF-IDF/similarity_data_TFIDF.csv')
    # test_data = pd.read_csv('/mnt/local/Baselines_Bugs/PatchFinder/data/test_data.csv')
    # test_data.drop(['desc_token', 'msg_token', 'diff_token'], axis=1, inplace=True)
    # tfidf_data = pd.merge(tf_idf_rawdata, test_data, on=['cve', 'owner', 'repo', 'commit_id', 'label'], how='right')
    # tfidf_data.to_csv(os.path.join(DATA_DIR, 'test_data_TFIDF.csv'), index=False)
    
    ### read tfidf similarity data
    print("Step 1/3: read tfidf similarity data")
    tfidf_data = pd.read_csv('/mnt/local/Baselines_Bugs/PatchFinder/TF-IDF/test_data_TFIDF.csv')
    tfidf_cve = tfidf_data.groupby('cve')

    print("Verify the number of cve:")
    cve_list = tfidf_data['cve'].unique().tolist()
    print('cve list length: {}'.format(len(cve_list)))
    cve_set = set(cve_list)
    print('cve set length: {}'.format(len(cve_set)))

    ### calculate the average rank of patch commit for each cve
    ### by using label to determine whether the commit is a patch commit
    print("Step 2/3: calculate the average rank of patch commit for each cve")
    rank_info = pd.DataFrame(columns=['cve', 'rank'])
    rank_info.to_csv(os.path.join(DATA_TMP_DIR, 'rankinfo_TDIDF.csv'), index=False)

    for cve, group in tqdm(tfidf_cve):
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
        rank_info_iter.to_csv(os.path.join(DATA_TMP_DIR, 'rankinfo_TDIDF.csv'), mode='a', header=False, index=False)
        # print('cve: {}, average rank: {}'.format(cve, average_rank))
    print('rank info saved')
    
    print("Step 3/3: calculate the recall, MRR and manual efforts")
    recall()
    mrr()
    manual_efforts()
    
    # # Verify the number of cases in each cve
    # print("Verify the number of cases in each cve")
    
    # num_cases_df = pd.DataFrame(columns=['cve', 'num_cases'])
    # num_cases_df.to_csv(os.path.join(DATA_TMP_DIR, 'num_cases_TFIDF.csv'), index=False)
    # for cve, group in tqdm(tfidf_cve):
    #     print('cve: {}, number of cases: {}'.format(cve, len(group)))
    #     num_cases_iter = pd.DataFrame([[cve, len(group)]], columns=['cve', 'num_cases'])
    #     num_cases_iter.to_csv(os.path.join(DATA_TMP_DIR, 'num_cases_TFIDF.csv'), mode='a', header=False, index=False)
    # print('number of cases info saved')
    
    
    # # Verify the number of cases within each cve in the test_data_top100.csv
    # test_data_top100 = pd.read_csv('/mnt/local/Baselines_Bugs/PatchFinder/data/test_data_top100.csv')
    
    # print("Verify the number of patch commits in each cve in the test_data_top100.csv")
    # test_data_top100_cve = test_data_top100.groupby('cve')
    
    # num_cases_df_top100 = pd.DataFrame(columns=['cve', 'num_cases'])
    # num_cases_df_top100.to_csv(os.path.join(DATA_TMP_DIR, 'num_cases_TFIDF_top100.csv'), index=False)
    # for cve, group in tqdm(test_data_top100_cve):
    #     print('cve: {}, number of cases: {}'.format(cve, len(group)))
    #     num_cases_iter = pd.DataFrame([[cve, len(group)]], columns=['cve', 'num_cases'])
    #     num_cases_iter.to_csv(os.path.join(DATA_TMP_DIR, 'num_cases_TFIDF_top100.csv'), mode='a', header=False, index=False)
    
    
    