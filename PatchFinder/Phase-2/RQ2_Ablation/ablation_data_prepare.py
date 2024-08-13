'''
ablation for only msg or only diff
'''
import os
import pandas as pd


msg_top100_dir= '/mnt/local/Baselines_Bugs/PatchFinder/TF-IDF/results/top_100_msg'
diff_top100_dir= '/mnt/local/Baselines_Bugs/PatchFinder/TF-IDF/results/top_100_diff'
os.makedirs(msg_top100_dir, exist_ok=True)
os.makedirs(diff_top100_dir, exist_ok=True)

source_dir = '/mnt/local/Baselines_Bugs/PatchFinder/TF-IDF/results/msg_diff'
DATA_DIR = '/mnt/local/Baselines_Bugs/PatchFinder/data'

file_list = ['similarity_data_msg_test.csv', 'similarity_data_msg_validate.csv', 'similarity_data_msg_train.csv']

data_list = ['test_data.csv', 'validate_data.csv', 'train_data.csv']

for data in data_list:
    data_path = os.path.join(DATA_DIR, data)
    suffix = data.split('_')[0]
    print("suffix: ", suffix)
    data_df = pd.read_csv(data_path)
    
    ### msg
    tfidf_msg_df = pd.read_csv(os.path.join(source_dir, 'similarity_data_msg_{}.csv'.format(suffix)))
    tfidf_msg_df_top100 = tfidf_msg_df.sort_values(by='similarity', ascending=False).groupby('cve').head(100)
    ## merge with data_df
    merged_msg_df = tfidf_msg_df_top100.merge(data_df, on=['cve', 'owner', 'repo', 'commit_id', 'label'])
    merged_msg_df.drop(columns=['diff_token'], inplace=True)
    merged_msg_df.to_csv(os.path.join(msg_top100_dir, 'similarity_data_msg_{}_top100.csv'.format(suffix)), index=False)
    print("merged_msg_df: ", merged_msg_df.shape)
    
    ### diff
    tfidf_diff_df = pd.read_csv(os.path.join(source_dir, 'similarity_data_diff_{}.csv'.format(suffix)))
    tfidf_diff_df_top100 = tfidf_diff_df.sort_values(by='similarity', ascending=False).groupby('cve').head(100)
    ## merge with data_df
    merged_diff_df = tfidf_diff_df_top100.merge(data_df, on=['cve', 'owner', 'repo', 'commit_id', 'label'])
    merged_diff_df.drop(columns=['msg_token'], inplace=True)
    merged_diff_df.to_csv(os.path.join(diff_top100_dir, 'similarity_data_diff_{}_top100.csv'.format(suffix)), index=False)
    print("merged_diff_df: ", merged_diff_df.shape)
    

    
    