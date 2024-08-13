import os
import pandas as pd

# test_cve_df = pd.read_csv('test_cve_info.csv')

# test_cve = test_cve_df['cve'].tolist()

# detected_cve_df = pd.read_csv('rank_info_top_10.csv')

# false_negative_cve = test_cve_df[~test_cve_df['cve'].isin(detected_cve_df['cve'])]
# print(false_negative_cve.shape)


# test_cve_text_df = pd.read_csv('/mnt/local/Baselines_Bugs/PatchFinder/data/test_data.csv')

# ## filter label =1
# test_cve_text_df = test_cve_text_df[test_cve_text_df['label']==1]

# print(test_cve_text_df.shape)

# ## filter only cve in test_cve
# test_cve_text_df = test_cve_text_df[test_cve_text_df['cve'].isin(test_cve)]

# print(test_cve_text_df.shape)


# ### merge with false negative cve
# false_negative_cve_text = pd.merge(false_negative_cve, test_cve_text_df, on='cve', how='left')

# print(false_negative_cve_text.shape)

# false_negative_cve_text.to_csv('false_negative_cve_text.csv', index=False)

fn_df = pd.read_csv('false_negative_cve_text.csv')

## count number of words in each cve description
fn_df['word_count'] = fn_df['desc_token'].apply(lambda x: len(str(x).split(" ")))

print(fn_df['word_count'].describe())

## print top 10 cve with lowest word count

print(fn_df.sort_values(by='word_count', ascending=True).head(30))

## count number of words in diff
fn_df['word_count'] = fn_df['diff_token'].apply(lambda x: len(str(x).split(" ")))

print(fn_df['word_count'].describe())

## print top 10 cve with highest word count

print(fn_df.sort_values(by='word_count', ascending=False).head(10))