import os
import logging
import torch
import pandas as pd
from collections import OrderedDict
from tqdm import tqdm
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
import configs
from load_data import CVEDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class CVEClassifier(pl.LightningModule):
    def __init__(self, 
                 lstm_hidden_size,
                 num_classes,
                 lstm_layers=1,
                 dropout=0.1,
                 lstm_input_size=512,
                 lr=5e-5,
                 num_train_epochs=20,
                 warmup_steps=1000,
                 ):
        
        super().__init__()
        self.codeReviewer = AutoModelForSeq2SeqLM.from_pretrained(
            "microsoft/codereviewer").encoder

        
        self.save_hyperparameters()
        
        # LSTM parameters
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_layers = lstm_layers
        self.dropout = dropout
        self.lstm_input_size = lstm_input_size
        self.criterion = nn.BCEWithLogitsLoss()

        self.desc_embedding = nn.Embedding(32216, lstm_input_size)
        '''
        vocab size: 32216
        https://huggingface.co/microsoft/codereviewer/blob/main/config.json#L94
        '''

        # LSTM layer
        self.lstm = nn.LSTM(input_size=self.lstm_input_size,
                            hidden_size=self.lstm_hidden_size,
                            num_layers=self.lstm_layers,
                            bidirectional=True,
                            batch_first=True)

        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)
        # Fully connected layer for output, and take care of the bidirectional
        self.fc = nn.Linear(2 * self.lstm_hidden_size + 2 * self.codeReviewer.config.hidden_size, num_classes)

    def forward(self, input_ids_desc, attention_mask_desc, input_ids_msg,
                attention_mask_msg, input_ids_diff, attention_mask_diff):
        
        # Getting embeddings for all inputs
        desc_embed = self.desc_embedding(input_ids_desc)
        
        # Pass through LSTM and max-pooling
        lstm_output, _ = self.lstm(desc_embed)
        max_pooled, _ = torch.max(lstm_output, 1)  # Max pooling
        
        
        # Get [CLS] embeddings for msg and diff
        msg_cls_embed = self.codeReviewer(input_ids=input_ids_msg, attention_mask=attention_mask_msg).last_hidden_state[:, 0, :]
        diff_cls_embed = self.codeReviewer(input_ids=input_ids_diff, attention_mask=attention_mask_diff).last_hidden_state[:, 0, :]
        
        # Concatenate max-pooled LSTM output and [CLS] embeddings
        concatenated = torch.cat((max_pooled, msg_cls_embed, diff_cls_embed), dim=1)
        
        # Apply dropout
        dropped = self.dropout_layer(concatenated)
        
        # Pass through the fully connected layer
        output = self.fc(dropped)
        
        return output
    
    def common_step(self, batch, batch_idx):
        
        predict = self(
            batch['input_ids_desc'],
            batch['attention_mask_desc'],
            batch['input_ids_msg'],
            batch['attention_mask_msg'],
            batch['input_ids_diff'],
            batch['attention_mask_diff']
        )
        # ValueError: Target size (torch.Size([512])) must be the same as input size (torch.Size([512, 1]))
        predict = predict.squeeze(1)
        loss = self.criterion(predict, batch['label'])
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        return loss

    def test_dataloader(self):
        return test_dataloader


def fix_state_dict(state_dict):
    """Prepend 'module.' to keys in state dictionary to match keys in model."""
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.' + k  
        new_state_dict[name] = v
    return new_state_dict

def load_checkpoint(load_path, model):
    """Load the model weights from a given path."""
    state_dict = torch.load(load_path)
    
    if load_path == '/mnt/local/Baselines_Bugs/PatchSleuth/model/output/Checkpoints/best-checkpoint.ckpt':
        fixed_state_dict = fix_state_dict(state_dict)
        model.load_state_dict(fixed_state_dict)
    else:
        model.load_state_dict(state_dict)

    return model

def save_outputs_to_csv(cve, output, label, data_path):
    """Save model outputs to a CSV file."""
    output_df = pd.DataFrame({'cve': [cve], 'output': [output], 'label': [label]})
    output_df.to_csv(data_path, mode='a', header=False, index=False)

def evaluate_single_batch(model, batch):
    """Evaluate model on a single batch and return the output."""
    device = configs.device
    model.to(device)

    input_keys = ['input_ids_desc', 'attention_mask_desc', 'input_ids_msg', 'attention_mask_msg', 'input_ids_diff', 'attention_mask_diff']
    inputs = [batch[key].to(device) for key in input_keys]

    return model(*inputs)

# Adding this function from the provided example
def save_metrics_to_csv(avg_recalls, avg_mrr, avg_manual_efforts, save_path):
    """Save recall, MRR, and manual efforts to a CSV file."""
    data = {
        'k': list(avg_recalls.keys()),
        'recall': [avg_recalls[k] for k in avg_recalls],
        'manual_effort': [avg_manual_efforts[k] for k in avg_recalls],
        'MRR': [avg_mrr for _ in avg_recalls]
    }
    df = pd.DataFrame(data)
    df.to_csv(save_path, index=False)

import pandas as pd

def compute_metrics(cve_data, k_values, rank_output_path):
    """Compute metrics (recall@k, MRR, Manual Efforts) for the model outputs."""
    recalls = {k: [] for k in k_values}
    mrrs = []
    manual_efforts = {k: [] for k in k_values}
    manual_efforts_sum = {k: 0 for k in k_values}
    manual_efforts_count = {k: 0 for k in k_values}
    
    
    
    rank_data = []

    for cve, data in cve_data.items():
        data.sort(key=lambda x: x[0], reverse=True)
        ranks = [(i + 1, label) for i, (_, label) in enumerate(data)]
        # Store ranks for later saving to CSV
        for rank, label in ranks:
            rank_data.append({'cve': cve, 'label': label, 'rank': rank})

        # Compute metrics using the ranks
        positive_ranks = [rank for rank, label in ranks if label == 1]
        for k in k_values:
            top_k_counts = sum(1 for rank in positive_ranks if rank < k)
            recalls[k].append(top_k_counts / len(positive_ranks) if positive_ranks else 0)
                
            if positive_ranks:
                min_rank_within_k = min([rank for rank in positive_ranks if rank < k], default=k)
            else:
                min_rank_within_k = k
                
            manual_efforts[k].append(min_rank_within_k)
            
            manual_efforts_sum[k] += min_rank_within_k
            manual_efforts_count[k] += 1
            
        reciprocal_ranks = [1 / (rank) for rank in positive_ranks]
        mrrs.append(sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0)

    avg_recalls = {k: sum(recalls[k]) / len(recalls[k]) if recalls[k] else 0 for k in k_values}
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0
    # avg_manual_efforts = {k: manual_efforts_sum[k] / manual_efforts_count[k] if manual_efforts_count[k] else 0 for k in k_values}
    avg_manual_efforts = {k: sum(manual_efforts[k]) / len(manual_efforts[k]) if manual_efforts[k] else 0 for k in k_values}
    # avg_manual_efforts[1] = 1.0
    
    # Save ranks to CSV using pandas
    if os.path.exists(rank_output_path):
        os.rename(rank_output_path, rank_output_path + '.bak')

    df = pd.DataFrame(rank_data)
    df.to_csv(rank_output_path, index=False)

    return avg_recalls, avg_mrr, avg_manual_efforts




def evaluate(model, testing_loader, k_values, reload_from_checkpoint=False, load_path_checkpoint=None, data_path='/mnt/local/Baselines_Bugs/PatchSleuth/metrics/CR_LSTM_0830/predict_final_model.csv', rank_info_path='/mnt/local/Baselines_Bugs/PatchSleuth/metrics/CR_LSTM_0830/ranks_final_model.csv'):
    """Evaluate the model on the given test loader."""
    # if reload_from_checkpoint:
    #     load_checkpoint(load_path_checkpoint, model)

    model.eval()
    cve_data = {}
    
    if os.path.exists(data_path):
        os.rename(data_path, data_path + '.bak')

    with torch.no_grad():
        for _, batch in tqdm(enumerate(testing_loader, 0), total=len(testing_loader)):
            output = evaluate_single_batch(model, batch)
            for cve_i, output_i, label_i in zip(batch['cve'], output, batch['label']):
                cve_data.setdefault(cve_i, []).append((output_i.item(), label_i.item()))
                save_outputs_to_csv(cve_i, output_i.item(), label_i.item(), data_path)

    avg_recalls, avg_mrr, manual_efforts = compute_metrics(cve_data, k_values, rank_info_path)

    for k in k_values:
        logging.info(f'Average Top@{k} recall: {avg_recalls[k]:.4f}')
        logging.info(f'Manual Effort@{k}: {manual_efforts[k]:.4f}')
    logging.info(f'Average MRR: {avg_mrr:.4f}')

    return avg_recalls, avg_mrr, manual_efforts

if __name__ == "__main__":
    MODEL_PATH = "/mnt/local/Baselines_Bugs/PatchSleuth/metrics/CR_LSTM_0830/Checkpoints/final_model.pt"  
    RANK_OUTPUT_PATH = "/mnt/local/Baselines_Bugs/PatchSleuth/metrics/CR_LSTM_0830/ranks_final_model.csv"
    test_data = CVEDataset(configs.test_file)
    test_dataloader = DataLoader(test_data, batch_size=4, num_workers=20)

    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100]
    # model_path = '/mnt/local/Baselines_Bugs/PatchSleuth/model/output/Checkpoints/best-checkpoint.ckpt'
    logging.info(f'Evaluating model at {MODEL_PATH}')
    
    # Instead of directly loading the model using torch.load, 
    # initialize it first and then load its state dictionary.
    model = CVEClassifier(
        lstm_hidden_size=256,
        num_classes=1,
        lstm_layers=1,
        dropout=0.1,
        lstm_input_size=512,
    )
    model.load_state_dict(torch.load(MODEL_PATH))

    
    data_path_flag = MODEL_PATH.split('/')[-1].split('.')[0]
    logging.info(f'data_path_flag: {data_path_flag}')
    
    recalls, avg_mrr, manual_efforts = evaluate(
        model, 
        test_dataloader, 
        k_values, 
        reload_from_checkpoint=True,
        load_path_checkpoint=MODEL_PATH,
        data_path=f'/mnt/local/Baselines_Bugs/PatchSleuth/metrics/CR_LSTM_0830/predict_{data_path_flag}.csv',
        rank_info_path=RANK_OUTPUT_PATH
    )

    # Save metrics to CSV
    metrics_save_path = f'/mnt/local/Baselines_Bugs/PatchSleuth/metrics/CR_LSTM_0830/metrics_{data_path_flag}.csv'
    if os.path.exists(metrics_save_path):
        os.rename(metrics_save_path, metrics_save_path + '.bak')
    
    save_metrics_to_csv(recalls, avg_mrr, manual_efforts, metrics_save_path)