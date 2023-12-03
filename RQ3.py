import os
import pandas as pd
from tqdm import tqdm
from cve_cne import get_nvd_info

DATA_DIR = '/mnt/local/Baselines_Bugs/PatchSleuth/TF-IDF/results/msg_diff'

file_list = ['similarity_data_msg_test.csv', 'similarity_data_msg_validate.csv', 'similarity_data_msg_train.csv']


# cve_repo_set = set()

# for file in file_list:
#     df = pd.read_csv(os.path.join(DATA_DIR, file), chunksize=10000)
#     for chunk in tqdm(df, desc="Reading data"):
#         cve_repos = chunk[['cve', 'owner', 'repo']].values.tolist()
#         cve_repo_set.update(tuple(x) for x in cve_repos)

# cve_repo_list = list(cve_repo_set)
# print("len(cve_repo_list): ", len(cve_repo_list))
# # save it into csv
# cve_repo_df = pd.DataFrame(cve_repo_list, columns=['cve', 'owner', 'repo'])
# cve_repo_df.to_csv("./cve_repo.csv", index=False)

# cve_repo_df = pd.read_csv("./cve_repo.csv")
# cve_repo_list = cve_repo_df['cve'].unique().tolist()
# print("len(cve_repo_list): ", len(cve_repo_list))

all_cve_info = []
# for cve, owner, repo in tqdm(cve_repo_list, desc="Getting CVE info"):
    # cve, links, cwe, cne, cvss2score = get_nvd_info(cve)
    # # print(cve, links, cwe, cne, cvss2score, owner, repo)
    # # save the info into csv
    # all_cve_info.append([cve, links, cwe, cne, cvss2score, owner, repo])

# all_cve_info_df = pd.DataFrame(all_cve_info, columns=['cve', 'links', 'cwe', 'cne', 'cvss2score'])
# all_cve_info_df.to_csv("./all_cve_info.csv", index=False, mode='w', header=True)

# for cve in tqdm(cve_repo_list, desc="Getting CVE info"):
#     cve, links, cwe, cne, cvss2score = get_nvd_info(cve)
#     # print(cve, links, cwe, cne, cvss2score, owner, repo)
#     # save the info into csv
#     tmp = pd.DataFrame([[cve, links, cwe, cne, cvss2score]], columns=['cve', 'links', 'cwe', 'cne', 'cvss2score'])
#     tmp.to_csv("./all_cve_info.csv", index=False, mode='a', header=False)

# test_df = pd.read_csv(os.path.join(DATA_DIR, file_list[0]))

# test_cve = test_df['cve'].unique().tolist()
# print("len(test_cve): ", len(test_cve))

# ## we need to get the cve info for the test_cve
# for cve in tqdm(test_cve, desc="Getting CVE info"):
#     cve, links, cwe, cne, cvss2score = get_nvd_info(cve)
#     # print(cve, links, cwe, cne, cvss2score, owner, repo)
#     # save the info into csv
#     tmp = pd.DataFrame([[cve, links, cwe, cne, cvss2score]], columns=['cve', 'links', 'cwe', 'cne', 'cvss2score'])
#     tmp.to_csv("./test_cve_info.csv", index=False, mode='a', header=False)
####################################################

test_cve_info_df = pd.read_csv("./test_cve_info_1.csv")
# ## we need to rescrawl the cve info for the test_cve for which we cannot get the cvss2score, which is `NVD-CVSS2-Other`

# for idx, row in tqdm(test_cve_info_df.iterrows(), desc="Getting CVE info", total=test_cve_info_df.shape[0]):
#     if row['cvss2'] == 'NVD-CVSS2-Other':
#         ## update the row
#         row['cvss2'] = get_nvd_info(row['cve'])[4]
#         test_cve_info_df.loc[idx] = row
# # update the test_cve_info.csv
# test_cve_info_df.to_csv("./test_cve_info.csv", index=False, mode='w', header=True)

###################################################

# rank_info_path = '/mnt/local/Baselines_Bugs/PatchSleuth/metrics/CR_0831/rank_info.csv'

# rank_df = pd.read_csv(rank_info_path)

# ## get the top 10 rank info for positive samples
# rank_df_top_10 = rank_df[rank_df['rank'] <= 10] # 10 is the top 10
# rank_df_top_10_pos = rank_df_top_10[rank_df_top_10['label'] == 1]
# print(rank_df_top_10_pos.shape)

# merged_df = rank_df_top_10_pos.merge(test_cve_info_df, on='cve', how='left')
# ## columns: cve,rank,links,cwe,cne,cvss2score
# merged_df.drop(columns=['links'], inplace=True, axis=1)
# merged_df.to_csv("./rank_info_top_10.csv", index=False, mode='w', header=True)

# rank_info_df = pd.read_csv("./rank_info_top_10.csv")
# cwe_dup = rank_info_df['cwe'].tolist()

# cwe = rank_info_df['cwe'].unique().tolist()
# print("len(cwe): ", len(cwe))

# "('CWE-601', ""URL Redirection to Untrusted Site ('Open Redirect')"")"

# cwe_ids = [x.split(',')[0].split("'")[1] for x in cwe]
# print("len(cwe_ids): ", len(cwe_ids))
# for cwe in cwe_ids:
#     if cwe.startswith('CWE-'):
#         # cwe = cwe[4:]
#         print(cwe)
        
# ## and we want to count the most frequent cwe in the top 10 rank info
# from collections import Counter
# cwe_counter = Counter(cwe_dup)
# print(cwe_counter.most_common(10)) # top 10

print("=========================================")
### count the most frequent cwe in the original test_cve_info.csv
test_cve_info_df = pd.read_csv("./test_cve_info.csv")
# cwe_dup_all = test_cve_info_df['cwe'].tolist()
# cwe_all_counter = Counter(cwe_dup_all)
# print(cwe_all_counter.most_common(10)) # top 10


# ###### we also need to count the least ratio of cwe in the top 10 rank info by using the cnt in top 10 rank info divide
# ###### the cnt in the original test_cve_info.csv
# cwe_ratio = []
# for cwe, cnt in cwe_counter.items():
#     cwe_ratio.append((cwe, cnt/cwe_all_counter[cwe], cnt, cwe_all_counter[cwe]))+

# cwe_ratio.sort(key=lambda x: x[1])
# print("=========================================")
# print(cwe_ratio[:10]) # top 10




import pandas as pd
from collections import Counter

# # Load the data
# rank_info_df = pd.read_csv("./rank_info_top_10.csv")
# test_cve_info_df = pd.read_csv("./test_cve_info_1.csv")

# # Step 1 & 2: Categorize the CVSS v2 scores into High, Medium, and Low severity
# def categorize_severity(score):
#     if isinstance(score, str) and score.startswith('NVD-CVSS2-'):
#         return 'Low'
#     elif isinstance(score, str):
#         score = float(score)
#         if score >= 7.0:
#             return 'High'
#         elif score >= 4.0:
#             return 'Medium'
#         else:
#             return 'Low'

# # Apply categorization
# rank_info_df['severity'] = rank_info_df['cvss2'].apply(categorize_severity)
# test_cve_info_df['severity'] = test_cve_info_df['cvss2'].apply(categorize_severity)

# # Step 3: Count the number of instances for each severity level in the top 10 ranks
# severity_counter_top_10 = Counter(rank_info_df['severity'])
# total_top_10 = len(rank_info_df)

# # Step 4: Calculate the percentages
# high_severity_percentage_top_10 = (severity_counter_top_10['High'] / total_top_10) * 100
# medium_severity_percentage_top_10 = (severity_counter_top_10['Medium'] / total_top_10) * 100
# low_severity_percentage_top_10 = (severity_counter_top_10['Low'] / total_top_10) * 100

# # Step 5: Observe the distribution of CVSS v2 scores in the original dataset
# severity_counter_all = Counter(test_cve_info_df['severity'])
# total_all = len(test_cve_info_df)

# # Step 6: Comparative analysis
# high_severity_percentage_all = (severity_counter_all['High'] / total_all) * 100
# medium_severity_percentage_all = (severity_counter_all['Medium'] / total_all) * 100
# low_severity_percentage_all = (severity_counter_all['Low'] / total_all) * 100

# print(f"High-severity in top 10: {high_severity_percentage_top_10}% ({severity_counter_top_10['High']}/{total_top_10})")
# print(f"Medium-severity in top 10: {medium_severity_percentage_top_10}% ({severity_counter_top_10['Medium']}/{total_top_10})")
# print(f"Low-severity in top 10: {low_severity_percentage_top_10}% ({severity_counter_top_10['Low']}/{total_top_10})")

# print(f"High-severity in all: {high_severity_percentage_all}% ({severity_counter_all['High']}/{total_all})")
# print(f"Medium-severity in all: {medium_severity_percentage_all}% ({severity_counter_all['Medium']}/{total_all})")
# print(f"Low-severity in all: {low_severity_percentage_all}% ({severity_counter_all['Low']}/{total_all})")


# # Step 7: Calculate the ratio for each severity level in the top 10 ranks to the total occurrences in the original dataset
# ratio_high_severity = (severity_counter_top_10['High'] / severity_counter_all['High']) * 100
# ratio_medium_severity = (severity_counter_top_10['Medium'] / severity_counter_all['Medium']) * 100
# ratio_low_severity = (severity_counter_top_10['Low'] / severity_counter_all['Low']) * 100

# print(f"Ratio of High-severity in top 10 to all: {ratio_high_severity}% ({severity_counter_top_10['High']}/{severity_counter_all['High']})")
# print(f"Ratio of Medium-severity in top 10 to all: {ratio_medium_severity}% ({severity_counter_top_10['Medium']}/{severity_counter_all['Medium']})")
# print(f"Ratio of Low-severity in top 10 to all: {ratio_low_severity}% ({severity_counter_top_10['Low']}/{severity_counter_all['Low']})")


#####################################################
## we need to analyze the owner and repo in the top 10 rank info, and the original test_cve_info.csv
rank_info_df = pd.read_csv("./rank_info_top_10.csv")

test_cve_info_df = pd.read_csv(os.path.join(DATA_DIR, file_list[0]))
test_cve_info_df.drop_duplicates(subset=['cve', 'owner', 'repo'], inplace=True)
## count the number of owner and repos total
# counter
owner_counter = Counter(test_cve_info_df['owner'].tolist())
repo_counter = Counter(test_cve_info_df['repo'].tolist())
print("owner_counter: ", owner_counter)
print("repo_counter: ", repo_counter)



merged_df = rank_info_df.merge(test_cve_info_df, on='cve', how='left')

merged_df.drop_duplicates(subset=['cve', 'owner', 'repo'], inplace=True)

## we only need the owner and repo
## counter
owner_counter_top_10 = Counter(merged_df['owner'].tolist())
repo_counter_top_10 = Counter(merged_df['repo'].tolist())

### calculate the ratio, and sort it

owner_ratio = []
repo_ratio = []

for owner, cnt in owner_counter_top_10.items():
    owner_ratio.append((owner, cnt/owner_counter[owner], cnt, owner_counter[owner]))

for repo, cnt in repo_counter_top_10.items():
    repo_ratio.append((repo, cnt/repo_counter[repo], cnt, repo_counter[repo]))


owner_ratio.sort(key=lambda x: x[1], reverse=True)
repo_ratio.sort(key=lambda x: x[1], reverse=True)

### print the top 10
# print("owner_ratio: ", owner_ratio[:10])
i = 0
for repo, ratio, cnt, cnt_all in repo_ratio:
    if cnt == 1:
        continue
    i += 1
    print(f"{repo}: {ratio} ({cnt}/{cnt_all})")
    if i == 20:
        break

print("=========================================")

#### also print the last 10
# print("owner_ratio: ", owner_ratio[-10:])
print("repo_ratio: ", repo_ratio[-10:])


'''
nanopb: 1.0 (2/2)
qemu: 1.0 (17/17)
krb5: 1.0 (2/2)
FFmpeg: 1.0 (27/27)
sqlite: 1.0 (2/2)
libgit2: 1.0 (3/3)
tor: 1.0 (2/2)
libming: 1.0 (2/2)
lxc: 1.0 (3/3)
ruby: 1.0 (2/2)
libtiff: 1.0 (2/2)
asylo: 1.0 (2/2)
rdesktop: 1.0 (4/4)
libxkbcommon: 1.0 (3/3)
oniguruma: 1.0 (2/2)
linux: 0.9743589743589743 (114/117)
wireshark: 0.9615384615384616 (25/26)
php-src: 0.8571428571428571 (6/7)
phpmyadmin: 0.8333333333333334 (10/12)
moodle: 0.8333333333333334 (5/6)
=========================================
repo_ratio:  [('jenkins', 0.5555555555555556, 5, 9), ('envoy', 0.5, 1, 2), ('minetest', 0.5, 1, 2), ('fbthrift', 0.5, 1, 2), ('imageworsener', 0.5, 1, 2), ('vim', 0.46153846153846156, 6, 13), ('radare2', 0.375, 3, 8), ('gpac', 0.25, 2, 8), ('OpenSC', 0.25, 1, 4), ('mruby', 0.2, 1, 5)]


'''