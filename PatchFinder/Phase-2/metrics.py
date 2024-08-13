'''
02/09/2023
To reuse the metrics calculation code in model/evaluate.py, we refactor the code to model/metrics.py.
This script is used to calculate Recall@k, MRR, and Manual Efforts@k for models.
'''

import logging
# from collections import OrderedDict
import pandas as pd

def compute_metrics(cve_data, k_values):
    '''
    Compute metrics: Recall@k, MRR and Manual Efforts@k.
    
    Parameters:
    - cve_data: dict
        Data with predicted values and actual labels.
    - k_values: list
        List of k values for which metrics are to be computed.
        
    Returns:
    - avg_recalls: dict
        Average recall values for different k values.
    - avg_mrr: float
        Average Mean Reciprocal Rank.
    - manual_efforts: dict
        Manual Efforts for different k values.
    '''
    
    recalls = {k: [] for k in k_values}
    mrrs = []
    manual_efforts = {k: [] for k in k_values}

    for _, data in cve_data.items():
        data.sort(key=lambda x: x[0], reverse=True)
        ranks = [i for i, (_, label) in enumerate(data) if label == 1]

        for k in k_values:
            top_k_counts = sum(1 for rank in ranks if rank < k)
            recalls[k].append(top_k_counts / len(ranks) if ranks else 0)

            effort_k = sum(min(rank, k) for rank in ranks) / len(ranks) if ranks else 0
            manual_efforts[k].append(effort_k)

        reciprocal_ranks = [1 / (rank + 1) for rank in ranks]
        mrrs.append(sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0)

    avg_recalls = {k: sum(recalls[k]) / len(recalls[k]) if recalls[k] else 0 for k in k_values}
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0

    return avg_recalls, avg_mrr, manual_efforts


def save_metrics_to_csv(avg_recalls, avg_mrr, manual_efforts, save_path):
    '''
    Save the computed metrics into a CSV file.
    
    Parameters:
    - avg_recalls: dict
        Average recall values for different k values.
    - avg_mrr: float
        Average Mean Reciprocal Rank.
    - manual_efforts: dict
        Manual Efforts for different k values.
    - save_path: str
        Path to save the CSV file.
    '''
    data = {
        'k': list(avg_recalls.keys()),
        'recall': [avg_recalls[k] for k in avg_recalls],
        'manual_effort': [sum(manual_efforts[k]) / len(manual_efforts[k]) if manual_efforts[k] else 0 for k in avg_recalls],
        'MRR': [avg_mrr for _ in avg_recalls]
    }
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)


'''
02/09/2023
To reuse the metrics calculation code in model/evaluate.py, we refactor the code to model/metrics.py.
This script is used to calculate Recall@k, MRR, and Manual Efforts@k for models.
'''

import logging
from collections import OrderedDict
import pandas as pd

def compute_metrics(cve_data, k_values):
    '''
    Compute metrics: Recall@k, MRR and Manual Efforts@k.
    
    Parameters:
    - cve_data: dict
        Data with predicted values and actual labels.
    - k_values: list
        List of k values for which metrics are to be computed.
        
    Returns:
    - avg_recalls: dict
        Average recall values for different k values.
    - avg_mrr: float
        Average Mean Reciprocal Rank.
    - manual_efforts: dict
        Manual Efforts for different k values.
    '''
    
    recalls = {k: [] for k in k_values}
    mrrs = []
    manual_efforts = {k: [] for k in k_values}

    for _, data in cve_data.items():
        data.sort(key=lambda x: x[0], reverse=True)
        ranks = [i for i, (_, label) in enumerate(data) if label == 1]

        for k in k_values:
            top_k_counts = sum(1 for rank in ranks if rank < k)
            recalls[k].append(top_k_counts / len(ranks) if ranks else 0)

            effort_k = sum(min(rank, k) for rank in ranks) / len(ranks) if ranks else 0
            manual_efforts[k].append(effort_k)

        reciprocal_ranks = [1 / (rank + 1) for rank in ranks]
        mrrs.append(sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0)

    avg_recalls = {k: sum(recalls[k]) / len(recalls[k]) if recalls[k] else 0 for k in k_values}
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0

    return avg_recalls, avg_mrr, manual_efforts


def save_predictions_to_csv(cve, output, label, data_path):
    '''
    Save model predictions to a CSV file.
    
    Parameters:
    - cve: list
        List of CVE identifiers.
    - output: list
        List of model predictions.
    - label: list
        Actual labels for each prediction.
    - data_path: str
        Path to save the CSV file.
    '''
    output_df = pd.DataFrame({'cve': cve, 'output': output, 'label': label})
    output_df.to_csv(data_path, mode='a', header=False, index=False)



def save_metrics_to_csv(avg_recalls, avg_mrr, manual_efforts, save_path):
    '''
    Save the computed metrics into a CSV file.
    
    Parameters:
    - avg_recalls: dict
        Average recall values for different k values.
    - avg_mrr: float
        Average Mean Reciprocal Rank.
    - manual_efforts: dict
        Manual Efforts for different k values.
    - save_path: str
        Path to save the CSV file.
    '''
    data = {
        'k': list(avg_recalls.keys()),
        'recall': [avg_recalls[k] for k in avg_recalls],
        'manual_effort': [sum(manual_efforts[k]) / len(manual_efforts[k]) if manual_efforts[k] else 0 for k in avg_recalls],
        'MRR': [avg_mrr for _ in avg_recalls]
    }
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)


def log_metrics(avg_recalls, avg_mrr, manual_efforts, k_values):
    '''
    Log the computed metrics.
    
    Parameters:
    - avg_recalls: dict
        Average recall values for different k values.
    - avg_mrr: float
        Average Mean Reciprocal Rank.
    - manual_efforts: dict
        Manual Efforts for different k values.
    - k_values: list
        List of k values for which metrics are to be computed.
    '''
    for k in k_values:
        logging.info(f'Average Top@{k} recall: {avg_recalls[k]:.4f}')
        avg_effort_k = sum(manual_efforts[k]) / len(manual_efforts[k]) if manual_efforts[k] else 0
        logging.info(f'Average Top@{k} manual efforts: {avg_effort_k:.4f}')
    logging.info(f'Average MRR: {avg_mrr:.4f}')

def print_metrics(avg_recalls, avg_mrr, manual_efforts, k_values):
    '''
    Print the computed metrics.
    
    Parameters:
    - avg_recalls: dict
        Average recall values for different k values.
    - avg_mrr: float
        Average Mean Reciprocal Rank.
    - manual_efforts: dict
        Manual Efforts for different k values.
    - k_values: list
        List of k values for which metrics are to be computed.
    '''
    for k in k_values:
        print(f'Average Top@{k} recall: {avg_recalls[k]:.4f}')
        avg_effort_k = sum(manual_efforts[k]) / len(manual_efforts[k]) if manual_efforts[k] else 0
        print(f'Average Top@{k} manual efforts: {avg_effort_k:.4f}')
    print(f'Average MRR: {avg_mrr:.4f}')