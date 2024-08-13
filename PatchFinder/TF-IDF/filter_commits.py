import pandas as pd
import os
from tqdm import tqdm

'''
Date: 24/08/2023
Purpose: Filter the commits by using TF-IDF score.
we try to filter only top 100 commits for each CVE according to the TF-IDF score.
'''

DATA_DIR = '/mnt/local/Baselines_Bugs/PatchFinder/data'
# ### columns in the three files: cve,owner,repo,commit_id,label,desc_token,msg_token,diff_token
# #  9.6G  test_data.csv
# #  84G   train_data.csv
# #  11G   validate_data.csv

tf_idf_df = pd.read_csv('/mnt/local/Baselines_Bugs/PatchFinder/TF-IDF/similarity_data_TFIDF.csv')
# cve,owner,repo,commit_id,similarity,label
# 2.0G Aug 24 12:18 similarity_data_TFIDF.csv

csv_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.csv')]

# Using tqdm to display progress
for f in tqdm(csv_files, desc="Processing files"):
    f_path = os.path.join(DATA_DIR, f)

    # Read the data file
    df = pd.read_csv(f_path)
    
    # Print the number of unique CVEs before filtering
    print(f"Number of unique CVEs in {f} before filtering: {df['cve'].nunique()}")
    
    # Merge with tf_idf_df
    merged_df = pd.merge(df, tf_idf_df, on=['cve', 'owner', 'repo', 'commit_id', 'label'], how='inner')
    
    # Sort the dataframe by similarity and filter top 100 commits for each CVE
    top100_df = merged_df.sort_values(by='similarity', ascending=False).groupby('cve').head(100)
    
    # Print the number of unique CVEs after filtering
    print(f"Number of unique CVEs in {f} after filtering: {top100_df['cve'].nunique()}")
    
    # Save the top 100 data into a new CSV
    output_file_name = f.split('.csv')[0] + "_top100.csv"
    output_path = os.path.join(DATA_DIR, output_file_name)
    top100_df.to_csv(output_path, index=False)

print("Filtering completed!")


