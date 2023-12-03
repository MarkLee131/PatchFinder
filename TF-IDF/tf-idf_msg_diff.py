# import nltk
import numpy as np
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import multiprocessing as mp
from tqdm import tqdm
# we need to import pkgs to count the duration
import time
# nltk.download('punkt')


# DATA_DIR = '/data/kaixuan/ramdisk/data'
# DATA_TMP_DIR = '/data/kaixuan/data_tmp'

DATA_DIR = '/mnt/local/Baselines_Bugs/PatchSleuth/data'

SAVE_DIR = '/mnt/local/Baselines_Bugs/PatchSleuth/TF-IDF/results/msg_diff'
os.makedirs(SAVE_DIR, exist_ok=True)

def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
                    
    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df



def compute_similarity(args):
    vectorizer = TfidfVectorizer()
    group, cve, prefix = args
    duration = []
    duration1 = []
    group['commits'] = group['msg_token']+ ' ' + group['diff_token']

    try:
        vectorizer.fit(group['commits'])
    except Exception as e:
        group['commits'].fillna(' ', inplace=True)
        vectorizer.fit(group['commits'])
    # Print the vocabulary size
    vocab_size = len(vectorizer.vocabulary_)
    # print(f'Vocabulary size: {vocab_size}')

    similarity_scores = []
    for _, row in group.iterrows():
        start = time.time()
        desc_tfidf = vectorizer.transform([row['desc_token']])
        combined_tfidf = vectorizer.transform([row['commits']])
        start1 = time.time()
        similarity = cosine_similarity(desc_tfidf, combined_tfidf).diagonal()[0]
        end = time.time()
        duration.append(end - start) ## this duration is for one commit
        duration1.append(end - start1) ## this duration is for one commit
        # print("duration: ", duration)
        # print("duration1: ", duration1)
        similarity_scores.append(similarity)
        
        
    # similarity_data = pd.DataFrame()
    # similarity_data['cve'] = group['cve']
    # similarity_data['owner'] = group['owner']
    # similarity_data['repo'] = group['repo']
    # similarity_data['commit_id'] = group['commit_id']
    # similarity_data['similarity'] = similarity_scores
    # similarity_data['label'] = group['label']

    # # Append to CSV
    # similarity_data.to_csv(os.path.join(SAVE_DIR, f'similarity_data_msg_{prefix}.csv'), mode='a', header=False, index=False)

    avg_duration = sum(duration)
    avg_duration1 = sum(duration1)
    print(avg_duration, avg_duration1)

    return avg_duration, avg_duration1
    

def compute_similarity_msg(args):
    vectorizer = TfidfVectorizer()
    group, cve, prefix = args

    try:
        vectorizer.fit(group['msg_token'])
    except Exception as e:
        group['msg_token'].fillna(' ', inplace=True)
        vectorizer.fit(group['msg_token'])
    # Print the vocabulary size
    vocab_size = len(vectorizer.vocabulary_)
    print(f'Vocabulary size: {vocab_size}')

    similarity_scores = []
    for _, row in group.iterrows():
        desc_tfidf = vectorizer.transform([row['desc_token']])
        combined_tfidf = vectorizer.transform([row['msg_token']])
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
    similarity_data.to_csv(os.path.join(SAVE_DIR, f'similarity_data_msg_{prefix}.csv'), mode='a', header=False, index=False)



def compute_similarity_diff(args):
    vectorizer = TfidfVectorizer()
    group, cve, prefix = args

    try:
        vectorizer.fit(group['diff_token'])
    except Exception as e:
        group['diff_token'].fillna(' ', inplace=True)
        vectorizer.fit(group['diff_token'])
    # Print the vocabulary size
    vocab_size = len(vectorizer.vocabulary_)
    print(f'Vocabulary size: {vocab_size}')

    similarity_scores = []
    for _, row in group.iterrows():
        desc_tfidf = vectorizer.transform([row['desc_token']])
        combined_tfidf = vectorizer.transform([row['diff_token']])
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
    similarity_data.to_csv(os.path.join(SAVE_DIR, f'similarity_data_diff_{prefix}.csv'), mode='a', header=False, index=False)





if __name__ == '__main__':
    
    # files = ['test_data.csv', 'validate_data.csv', 'train_data.csv']
    # Load data
    files = ['test_data.csv']
    
    for file in files:
        print("Processing file: ", file)
        prefix = file.split('_')[0]
        
        # # Create and write the header of the CSV file
        # empty_df = pd.DataFrame(columns=['cve', 'owner', 'repo', 'commit_id', 'similarity', 'label'])
        # empty_df.to_csv(os.path.join(SAVE_DIR, f'similarity_data_msg_{prefix}.csv'), index=False)
        
        # # empty_df = pd.DataFrame(columns=['cve', 'owner', 'repo', 'commit_id', 'similarity', 'label'])
        # empty_df.to_csv(os.path.join(SAVE_DIR, f'similarity_data_diff_{prefix}.csv'), index=False)
        
        
        
        data = pd.read_csv(os.path.join(DATA_DIR, file))

    
        ### load the tokenized data
        data['desc_token'] = data['desc_token'].fillna(' ')
        data['msg_token'] = data['msg_token'].fillna(' ')
        data['diff_token'] = data['diff_token'].fillna(' ')

        ### concat the tokenized data

        # Combine tokenized commit messages and diffs

        # data['combined'] = data['msg_token'] + ' ' + train_data['diff_token']

        
        # Create a multiprocessing pool
        pool = mp.Pool(mp.cpu_count())
        
        data_cve = data.groupby('cve')
        cve_list = data['cve'].unique()
        print("len(cve_list): ", len(cve_list))
        
        # Process each chunk independently
        print("Computing TF-IDF vectors...")
        
        
        [durations, durations1] = list(tqdm(pool.imap_unordered(compute_similarity, [(group, cve, prefix) for cve, group in data_cve]), total=len(cve_list), desc="Computing similarity scores for msg"))
        avg_duration = sum(durations) / len(durations)
        avg_duration1 = sum(durations1) / len(durations1)
        print("avg_duration: ", avg_duration)
        print("avg_duration1: ", avg_duration1)
        break       
        
        
        results_msgs  = list(tqdm(pool.imap_unordered(compute_similarity_msg, [(group, cve, prefix) for cve, group in data_cve]), total=len(cve_list), desc="Computing similarity scores for msg"))

        print("Saved similarity_data.csv to {}".format(os.path.join(SAVE_DIR, f'similarity_data_msg_{prefix}.csv')))

        results_diff = list(tqdm(pool.imap_unordered(compute_similarity_diff, [(group, cve, prefix) for cve, group in data_cve]), total=len(cve_list), desc="Computing similarity scores for diff"))
        
        print("Saved similarity_data.csv to {}".format(os.path.join(SAVE_DIR, f'similarity_data_diff_{prefix}.csv')))