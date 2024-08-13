# Data Preparation

In this folder, you are supposed to prepare the **21,781,044** commits used in the paper, and the CVE descriptions from the NVD database.


>To create a robust training set, we followed the practices in prior works [38, 42] to sample 5,000 non-patch commits as negative samples for each CVE. However, in scenarios where a repository contained fewer than 5,000 commits in total, we included all available non-patch commits as negative samples. Finally, we got **21,781,044** commits in total. 

Since the data consumes a large amount of space (110GB), we provide a sample dataset in the `sample` folder and describe the core logic of the data preparation process.

## Data Preparation Process

As mentioned in the paper, we collected the commits from the repositories that contain the 4870 patch commits. The data preparation process is as follows:

1. You can first `git clone` the repositories that contain the 4870 commits:

(1) The detailed information of these patch commits can be found [./patch_data_4780.csv](./patch_data.csv).

(2) By using the `owner` and `repo`, you can clone the repositories and checkout to the specific commit.



>Note: Since some commits maybe unavailable due to the branch change, we found there are some useful strategies to handle this issue:

(a) cd to the repository folder and run `git checkout <commit_id>` before running the `git log` command.

(b) We also use GitHub APIs to sumplement the missing commits, but it is not recommended due to the rate limit.



2. For non-patch commits, we randomly sampled the other 5,000 commits from the same repositories. 

For your convenience, we suggest sampling the non-patch commits from local repositories by using `git log` command. 

>In our case:

(1) We first used `git log` and GitHub API to get a 20k commit list, and then randomly sampled 5,000 commits as negative samples (removing the patch commits if any).

(2) Then, we used `git show` to get the commit message, author, and diff information.

(3) Finally, we saved the data in the format of `commit_id, commit_message, author, diff` in the `non_patch_data.csv` file.

3. Since the raw data is too large, especially the diff information, which is difficult to feed into the model directly, we cleaned the data by removing the unnecessary information including empty lines, lines without any code changes, and only keep the first 1,000 lines. Note that do not remove the lines starting with `@@` since they are necessary for the model to understand the context of the code changes.

4. The above process is for commits, and we also need to prepare CVE data. We collected the CVE descriptions from the NVD database and saved them in a column of a csv file. 

5. Based on the cleaned data, we tried to tokenize the data (CVE description and commits) by using `NLTK` library to make it more readable for the model. 
We need to mention that we also tried to tokenize them by using programming language-specific tokenizers like ANTLR, but the results were not as good as using the `NLTK` library since ANTLR cannot handle the code changes well, especially in our case where the code starts with `@@`, `+`, `-`, etc.

6. Finally, we saved the data in the format of `cve,owner,repo,commit_id,label,desc_token,msg_token,diff_token` in the `data.csv` file. By splitting the data into training, validation, and test sets (80%, 10%, 10% according to the CVEs), we got the final data for the model training. The data was finally saved in the `data` folder Their names are `train_data.csv`, `validate_data.csv`, and `test_data.csv`.

These three files are not included in the repository due to the large size, but we can provide a sample dataset with 5,001 commits in the `sample` folder. 




