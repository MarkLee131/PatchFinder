
# Trails for GPT-4 & CodeLlama

This folder contains the scripts and results of the trials conducted with prompting CodeLlama for the task of patch commit identification.

```bash
.
|-- README.md
|-- codellama_batch.py # Script for prompting CodeLlama
|-- codellama_results_sample.zip # Zipped results of prompting CodeLlama
|-- configs.py # Configuration file for prompting CodeLlama
`-- retrieval # Can be ignored since we zipped the results to codellama_results_sample.zip
    |-- gpu_2
    `-- gpu_3
```


## Detailed Description

We also considered the capability of popular LLMs including GPT-4 and CodeLlama for this task. 

Regarding GPT-4, while its capabilities in Automatic Program Repair are notable as shown in ChatRepair [1], its application in our large-scale analysis (21,781,044 <CVE description, code commit> pairs in total) faced considerable financial cost [2]. 

For a clearer understanding, let's consider a quantitative analysis of the financial cost involved in using GPT-4 for our task. Each CVE analysis requires processing a significant number of commits, averaging around 5000. Given the token limit (combining 128 tokens for the CVE description and 512 tokens for each commit), the cost calculation would be as follows: 

```
Cost per CVE=(128 tokens (CVE)+512 tokens (Commit))×5000 commits/1000 tokens×$0.03/1000= $9.6


*Note: This is only the cost of the prompt tokens ($0.03/1k prompt tokens) based on 8k context, where the sampled token price ($0.06/1k sampled tokens) is not yet included. 

```

Considering the significant financial expenses involved, we believe it is not a good choice for practical use. Also, the large amount of these commits makes it impossible for users to manually query ChatGPT. All these factors underscore the impracticality of employing ChatGPT in this task.

In our trials with CodeLlama, we encountered challenges with efficiency and effectiveness. For a practical test involving 20 sampled CVEs, `CodeLlama-7b-Instruct-hf` required over two days (**2 days, 5 hours, 27 minutes, and 30 seconds**) to process 19,649 commits, covering **less than 4 CVEs**. We have uploaded [the results of prompting CodeLlama](./codellama_results_sample.zip) as well. This inefficiency, coupled with suboptimal results, led us to reconsider its suitability for our task: Locate the patch commit from around 5000 commits for each CVE. 

Ultimately, we chose CodeReviewer for PatchFinder, aiming for a balance between effective patch tracing and practical considerations such as processing time and cost. This approach has allowed us to address the unique challenges in tracing security patches for CVEs in an efficient and effective manner. 

Notably, PatchFinder achieves a Recall@10 of 80.63%, an MRR of 0.7951, and the Manual Effort@10 required is curtailed to 2.77, marking a 1.94 times improvement over current leading methods. It shows that CodeReviewer (220M parameters) has already achieved good performance on ranking the commits, hence there is little gain to consider a billion-level LLM like CodeLlama. 


[1] Xia, Chunqiu Steven, and Lingming Zhang. "Keep the Conversation Going: Fixing 162 out of 337 Bugs for $0.42 Each Using ChatGPT." arXiv preprint arXiv:2304.00385 (2023). 

[2] https://openai.com/pricing.