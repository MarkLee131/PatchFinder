# About this folder

## Folder structure
```bash
tmp_0830 ## the default folder to save the metrics (used by `tf-idf_metrics.py`)
README.md 
ablation_get_top100_results.py ## for ablation usage
ablation_tf-idf_msg_diff.py    ## for ablation usage
split_data_data_preparation.py ## for data preparation only, to split the data (20 million entries you collect)
tf-idf_calculate_similarity.py ## used by lexical-based retriever. run it first.
tf-idf_metrics.py              ## used by lexical-based retriever. run it after the `tf-idf_calculate_similarity.py `
```

## lexical-based retriever

To run it, we assume that you have prepared the evaluation data, and now we calculate the tf-idf scores, then rank them.

1. Run `tf-idf_calculate_similarity.py` first to get the tf-idf scores.

2. Run `tf-idf_metrics.py` to rank the commits, and get the metrics for them. The results will be saved into `tmp_0830` by default.


