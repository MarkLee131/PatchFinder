import os
import pandas as pd
from rank_bm25 import BM25Okapi, BM25, BM25L, BM25Plus
import gc
import multiprocessing as mp
from tqdm import tqdm
import datetime
import time

DATA_DIR = '/home/kaixuan/patch_locating/data/split_data'
SAVE_DIR = '/home/kaixuan/patch_locating/PatchSleuth/BM25'

train_file = os.path.join(DATA_DIR, 'train_data.csv')
test_file = os.path.join(DATA_DIR, 'test_data.csv')
validate_file = os.path.join(DATA_DIR, 'validate_data.csv')

# def filter_large_groups(group):
#     if len(group) > 5000:
#         positive_rows = group[group['label'] == 1]
#         negative_rows = group[group['label'] == 0].sample(n=4999)
#         return pd.concat([positive_rows, negative_rows])
#     return group

def compute_similarity(args):
    duration = []
    group, cve = args
    # group = filter_large_groups(group)

    tokenized_data = [doc.split() for doc in group['combined']]
    bm25 = BM25Okapi(tokenized_data, k1=1.5, b=0.5)
    # bm25 = BM25Plus(tokenized_data, k1=1.6, b=0.3, delta=0.7)
    # bm25 = BM25L(tokenized_data, k1=1.8, b=0.3, delta=0.6)

    start = time.time()
    similarity_scores = bm25.get_scores(group['desc_token'].iloc[0].split())
    end = time.time()
    duration.append(end - start)
    assert len(similarity_scores) == len(group), "Length of similarity scores does not match length of group"
    
    ## detect whether the values in similarity_scores are negative

    
    with open(os.path.join(SAVE_DIR, 'similarity_data_bm25okapi.csv'), 'a') as f:
        i = 0
        for index, row in group.iterrows():
            f.write(f"{row['cve']},{row['owner']},{row['repo']},{row['commit_id']},{similarity_scores[i]},{row['label']}\n")
            i += 1
            
    duration = sum(duration) / len(duration)
    print(f"Average time to compute similarity scores for {cve}: {duration}")
    return duration

def process_full_dataframe(df):
    df['desc_token'].fillna(' ', inplace=True)
    df['combined'] = df['msg_token'].fillna(' ') + ' ' + df['diff_token'].fillna(' ')
    df.drop(columns=['msg_token', 'diff_token'], inplace=True)
    
    data_cve = df.groupby('cve')
    cve_list = df['cve'].unique()

    pool = mp.Pool(mp.cpu_count())
    durations = list(tqdm(pool.imap_unordered(compute_similarity, [(group, cve) for cve, group in data_cve]), total=len(cve_list)))
    pool.close()
    pool.join()
    avg_duration = sum(durations) / len(durations)
    print(f"Average time to compute similarity scores: {avg_duration}")

def main():
    similarity_data_file = os.path.join(SAVE_DIR, 'similarity_data_bm25okapi.csv')
    print("similarity_data_file: ", similarity_data_file)
    print("Start time: ", datetime.datetime.now())

    with open(similarity_data_file, 'w') as f:
        f.write('cve,owner,repo,commit_id,similarity,label\n')
    
    print(f"Processing: {test_file}")
    df = pd.read_csv(test_file)
    process_full_dataframe(df)
    del df
    gc.collect()

    print("Saved similarity_data.csv to {}".format(os.path.join(SAVE_DIR, 'similarity_data_bm25okapi.csv')))

# Run the main function
main()
