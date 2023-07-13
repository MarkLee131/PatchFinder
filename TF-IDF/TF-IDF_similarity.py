# import nltk
# import numpy as np
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import reduce_mem_usage
from tqdm import tqdm
import gc
import multiprocessing as mp
from tqdm import tqdm
# nltk.download('punkt')

##### 2023.07.07 
##### This script is revised from TFIDF_similarity_old.py
##### We changed the logic to first save the tokenized data to csv files, and then load them to save memory.

##### 2023.07.09
##### We try to paralellize the process of calculating similarity scores, since the single process is too slow (17hs).

DATA_DIR = '/data/kaixuan/ramdisk/data'
DATA_TMP_DIR = '/data/kaixuan/data_tmp'

# scp -P 32908 kaixuan_cuda11@XXXXXX:/mnt/local/Baselines_Bugs/CodeBert/data/cve_desc.csv ./data/

# sudo mkdir -p /data/kaixuan/ramdisk
# sudo mount -t tmpfs -o size=110G tmpfs /data/kaixuan/ramdisk
# mv ./data /data/kaixuan/ramdisk/

# sync
# sudo umount /data/kaixuan/ramdisk



def compute_similarity(args):
    vectorizer = TfidfVectorizer()
    group, cve = args

    try:
        vectorizer.fit(group['combined'])
    except Exception as e:
        group['combined'].fillna('', inplace=True)
        vectorizer.fit(group['combined'])

    similarity_scores = []
    for _, row in group.iterrows():
        desc_tfidf = vectorizer.transform([row['desc_token']])
        combined_tfidf = vectorizer.transform([row['combined']])
        similarity = cosine_similarity(desc_tfidf, combined_tfidf).diagonal()[0]
        similarity_scores.append(similarity)

    similarity_data = pd.DataFrame()
    similarity_data['cve'] = group['cve']
    similarity_data['owner'] = group['owner']
    similarity_data['repo'] = group['repo']
    similarity_data['commit_id'] = group['commit_id']
    similarity_data['similarity'] = similarity_scores
    similarity_data['label'] = group['label']

    # Append to CSV
    similarity_data.to_csv(os.path.join(DATA_DIR, 'similarity_data_1.csv'), mode='a', header=False, index=False)


if __name__ == '__main__':
    
    # Create and write the header of the CSV file
    empty_df = pd.DataFrame(columns=['cve', 'owner', 'repo', 'commit_id', 'similarity', 'label'])
    empty_df.to_csv(os.path.join(DATA_DIR, 'similarity_data.csv'), index=False)


    # Load data
    # commit_data = pd.read_csv(os.path.join(DATA_DIR, 'commit_sample.csv')) ### for test
    commit_data = pd.read_csv(os.path.join(DATA_DIR, 'commit_info.csv'))
    reduce_mem_usage(commit_data)
    desc_data = pd.read_csv(os.path.join(DATA_DIR, 'cve_desc.csv'))

    # Merge commit_data and desc_data on 'cve' column
    data = pd.merge(commit_data, desc_data, on='cve', how='left')

    # Reduce memory usage
    reduce_mem_usage(data)

    data = data.drop(columns=['cve_desc', 'msg', 'diff'])

    del commit_data, desc_data
    gc.collect()
    
    ### load the tokenized data
    print("Loading tokenized data...")
    desc_df = pd.read_csv(os.path.join(DATA_TMP_DIR, 'desc_token.csv'))
    reduce_mem_usage(desc_df)
    desc_df['desc_token'] = desc_df['desc_token'].fillna(' ') # replace NaN values with a space
    
    msg_df = pd.read_csv(os.path.join(DATA_TMP_DIR, 'msg_token.csv'))
    reduce_mem_usage(msg_df)
    msg_df['msg_token'] = msg_df['msg_token'].fillna(' ') # replace NaN values with a space
    
    diff_df = pd.read_csv(os.path.join(DATA_TMP_DIR, 'diff_token.csv'))
    reduce_mem_usage(diff_df)
    diff_df['diff_token'] = diff_df['diff_token'].fillna(' ') # replace NaN values with a space

    # print("shape of desc_df: ", desc_df.shape)
    # print("shape of msg_df: ", msg_df.shape)
    # print("shape of diff_df: ", diff_df.shape)

    ### concat the tokenized data

    # Combine tokenized commit messages and diffs

    data['combined'] = msg_df['msg_token'] + " " + diff_df['diff_token']
    print("shape of data: ", data.shape)
    print("data['combined'][0]: ", data['combined'][0])

    del msg_df, diff_df
    gc.collect()

    data = pd.concat([data, desc_df['desc_token']], axis=1)

    
    # Create a multiprocessing pool
    pool = mp.Pool(mp.cpu_count())
    
    data_cve = data.groupby('cve')
    cve_list = data['cve'].unique()
    print("len(cve_list): ", len(cve_list))
    
    # Process each chunk independently
    print("Computing TF-IDF vectors...")
    results = list(tqdm(pool.imap_unordered(compute_similarity, [(group, cve) for cve, group in data_cve]), total=len(cve_list), desc="Computing similarity scores"))
    
    print("Saved similarity_data.csv to {}".format(os.path.join(DATA_DIR, 'similarity_data.csv')))
