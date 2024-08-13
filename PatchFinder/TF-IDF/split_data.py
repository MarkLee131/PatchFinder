# import nltk
# import numpy as np
import os
import pandas as pd

import sys
sys.path.append("..") # adds the parent directory to Python's search path
from BM25.utils import reduce_mem_usage

import gc
import numpy as np

##### 2023.07.18
##### Used to split the data for training and testing

DATA_DIR = '/data/kaixuan/ramdisk/data'
DATA_TMP_DIR = '/data/kaixuan/data_tmp'
SPLIT_DATA_DIR = '/data/kaixuan/data_tmp/split_data'
os.makedirs(SPLIT_DATA_DIR, exist_ok=True)

if __name__ == '__main__':

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


    data = pd.concat([data, desc_df['desc_token'], msg_df['msg_token'], diff_df['diff_token']], axis=1)

    
    data_cve = data.groupby('cve')
    cve_list = data['cve'].unique()
    print("len(cve_list): ", len(cve_list))
    
    # split the data into train, validation and test by setting the random seed 3407
    np.random.seed(3407)
    train_cve = np.random.choice(cve_list, size=int(len(cve_list)*0.8), replace=False)
    # save the train_cve groups into a csv file
    train_df = data[data['cve'].isin(train_cve)]
    train_df.to_csv(os.path.join(SPLIT_DATA_DIR, 'train_data.csv'), index=False)
    print(f"Number of unique 'cve' in train_data: {train_df['cve'].nunique()}")

    validate_cve = np.random.choice(list(set(cve_list)-set(train_cve)), size=int(len(cve_list)*0.1), replace=False)
    # save the validate_cve groups into a csv file
    validate_df = data[data['cve'].isin(validate_cve)]
    validate_df.to_csv(os.path.join(SPLIT_DATA_DIR, 'validate_data.csv'), index=False)
    print(f"Number of unique 'cve' in validate_data: {validate_df['cve'].nunique()}")

    test_cve = list(set(cve_list)-set(train_cve)-set(validate_cve))
    # save the test_cve groups into a csv file
    test_df = data[data['cve'].isin(test_cve)]
    test_df.to_csv(os.path.join(SPLIT_DATA_DIR, 'test_data.csv'), index=False)
    print(f"Number of unique 'cve' in test_data: {test_df['cve'].nunique()}")

# len(cve_list):  4789
# Number of unique 'cve' in train_data: 3831
# Number of unique 'cve' in validate_data: 478
# Number of unique 'cve' in test_data: 480
## scp -r ./split_data/ -P xxx kaixuan_cuda11@xxxxxx:/mnt/local/Baselines_Bugs/PatchFinder/data/