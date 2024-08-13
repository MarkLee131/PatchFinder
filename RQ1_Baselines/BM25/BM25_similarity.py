# import nltk
# import numpy as np
import os
import pandas as pd
from rank_bm25 import BM25Okapi
from utils import reduce_mem_usage
from tqdm import tqdm
import gc
import multiprocessing as mp
from tqdm import tqdm

##### 2023.07.13
'''
We replace TF-IDF by using BM25 to calculate the similarity 
between the description of cve and the commit message of commit.
'''

DATA_DIR = '/data/kaixuan/ramdisk/data'
DATA_TMP_DIR = '/data/kaixuan/data_tmp'

def compute_similarity(args):
    group, cve = args

    corpus = [text.lower().split() for text in group['combined'].fillna('').tolist()]  # Ensure no NaN values in corpus, convert to list of tokens
    # use ATIRE BM25, and the best parameters for ATIRE BM25 are b=0.3 and k1=1.1 
    # bm25 = BM25Okapi(corpus, k1=1.1, b=0.3)
    bm25 = BM25Okapi(corpus)

    query = group['desc_token'].iloc[0]
    scores = bm25.get_scores(query.split())  # Tokenize query, BM25 assumes tokenized input
    # top_50 = bm25.get_top_n(query.split(), corpus, n=50)

    similarity_data = pd.DataFrame()
    similarity_data['cve'] = group['cve']
    similarity_data['owner'] = group['owner']
    similarity_data['repo'] = group['repo']
    similarity_data['commit_id'] = group['commit_id']
    similarity_data['similarity'] = scores
    similarity_data['label'] = group['label']

    # Sort by similarity scores
    similarity_data = similarity_data.sort_values('similarity', ascending=False)

    # Append to CSV
    similarity_data.to_csv(os.path.join(DATA_DIR, 'similarity_data_bm25.csv'), mode='a', header=False, index=False)


if __name__ == '__main__':
    
    # Create and write the header of the CSV file
    empty_df = pd.DataFrame(columns=['cve', 'owner', 'repo', 'commit_id', 'similarity', 'label'])
    
    ### if the file exists, rename it to .bak
    if os.path.exists(os.path.join(DATA_DIR, 'similarity_data_bm25.csv')):
        os.rename(os.path.join(DATA_DIR, 'similarity_data_bm25.csv'), os.path.join(DATA_DIR, 'similarity_data_bm25.csv.bak'))
    
    empty_df.to_csv(os.path.join(DATA_DIR, 'similarity_data_bm25.csv'), index=False)

    # Load data
    # commit_data = pd.read_csv(os.path.join(DATA_DIR, 'commit_sample.csv')) ### for test
    commit_data = pd.read_csv(os.path.join(DATA_DIR, 'commit_info.csv'))
    reduce_mem_usage(commit_data)
    print("shape of commit_data: ", commit_data.shape)
    desc_data = pd.read_csv(os.path.join(DATA_DIR, 'cve_desc.csv'))
    print("shape of desc_data: ", desc_data.shape)

    # Merge commit_data and desc_data on 'cve' column
    data = pd.merge(commit_data, desc_data, on='cve', how='left')
    print("shape of data: ", data.shape)

    # Reduce memory usage
    reduce_mem_usage(data)

    data = data.drop(columns=['cve_desc', 'msg', 'diff'])

    del commit_data, desc_data
    gc.collect()
    
    ### load the tokenized data
    print("Loading tokenized data...")
    desc_df = pd.read_csv(os.path.join(DATA_TMP_DIR, 'desc_token.csv'))
    reduce_mem_usage(desc_df)
    
    # replace NaN values with a space, and convert to lowercase
    desc_df['desc_token'] = desc_df['desc_token'].fillna(' ').str.lower()
    
    msg_df = pd.read_csv(os.path.join(DATA_TMP_DIR, 'msg_token.csv'))
    reduce_mem_usage(msg_df)
    
    msg_df['msg_token'] = msg_df['msg_token'].fillna(' ').str.lower() # replace NaN values with a space
    
    diff_df = pd.read_csv(os.path.join(DATA_TMP_DIR, 'diff_token.csv'))
    reduce_mem_usage(diff_df)
    diff_df['diff_token'] = diff_df['diff_token'].fillna(' ').str.lower() # replace NaN values with a space

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
    #### calculate the groups of cve
    len_group = len(data_cve)
    cve_list = data['cve'].unique()
    print("len(cve_list): ", len(cve_list))

    # Process each chunk independently
    print("Computing BM25 vectors...")
    # results = list(tqdm(pool.imap_unordered(compute_similarity, [(group, cve) for cve, group in data_cve]), total=len_group, desc="Computing similarity scores"))
    for _ in tqdm(pool.imap_unordered(compute_similarity, [(group, cve) for cve, group in data_cve]), total=len_group, desc="Computing similarity scores"):
        pass

    pool.close()
    pool.join()

    
    print("Saved similarity_data_bm25.csv to {}".format(os.path.join(DATA_DIR, 'similarity_data_bm25.csv')))
