from collections import OrderedDict
import torch
import numpy as np
import logging
from load_data import CVEDataset
from torch.utils.data import DataLoader
import configs
import csv
# from main_new import CVEClassifier

from transformers import AutoModelForSeq2SeqLM, AdamW, get_linear_schedule_with_warmup
import pytorch_lightning as pl
import torch.nn as nn
import torch
# import torch.optim.AdamW as AdamW

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
        
        ### This is default setting.
        # # Set requires_grad=True for all parameters of codeReviewer to fine-tune it
        # for param in self.codeReviewer.parameters():
        #     param.requires_grad = True

        
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

    # def training_step(self, batch, batch_idx):
    #     loss = self.common_step(batch, batch_idx)
    #     # logs metrics for each training_step,
    #     # and the average across the epoch
    #     self.log("training_loss", loss)
    #     return loss

    # def validation_step(self, batch, batch_idx):
    #     loss = self.common_step(batch, batch_idx)
    #     self.log("validation_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

    #     return loss

    def test_step(self, batch, batch_idx):
        loss = self.common_step(batch, batch_idx)
        return loss

    # def configure_optimizers(self):
    #     # create optimizer
    #     optimizer = AdamW(self.parameters(), lr=self.hparams.lr)
    #     # create learning rate scheduler
    #     num_train_optimization_steps = self.hparams.num_train_epochs * len(train_dataloader)
    #     lr_scheduler = {'scheduler': get_linear_schedule_with_warmup(optimizer,
    #                                                 num_warmup_steps=self.hparams.warmup_steps,
    #                                                 num_training_steps=num_train_optimization_steps),
    #                     'name': 'learning_rate',
    #                     'interval':'step',
    #                     'frequency': 1}

    #     return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    # def train_dataloader(self):
    #     return train_dataloader

    # def val_dataloader(self):
    #     return valid_dataloader

    def test_dataloader(self):
        return test_dataloader






logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def fix_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = 'module.' + k  # add `module.` prefix
        new_state_dict[name] = v
    return new_state_dict



def load_checkpoint(load_path, model):
    
    # Load the saved state dict
    state_dict = torch.load(load_path)
    
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    print("\n\nLoaded state_dict:")
    for param_tensor in state_dict:
        print(param_tensor, "\t", state_dict[param_tensor].size())
    
    if load_path == '/mnt/local/Baselines_Bugs/PatchSleuth/model/output/Checkpoints/best-checkpoint.ckpt':
        fixed_state_dict = fix_state_dict(state_dict)
        model.load_state_dict(fixed_state_dict)
        print("\n\nFixed state_dict:")
        for param_tensor in fixed_state_dict:
            print(param_tensor, "\t", fixed_state_dict[param_tensor].size())
    else:
        assert load_path == '/mnt/local/Baselines_Bugs/PatchSleuth/model/output/Checkpoints/final_model.pt'
        model.load_state_dict(state_dict)

    return model


def top_k_recall(y_true, y_pred, k):
    assert len(y_true) == len(y_pred)
    total_relevant = sum(y_true)
    hit_count = sum(y_pred[:k])
    return hit_count / total_relevant

def mean_reciprocal_rank(y_true, y_pred):
    assert len(y_true) == len(y_pred)
    mrr = 0.0
    for true, pred in zip(y_true, y_pred):
        for idx, item in enumerate(pred):
            if item == true:
                mrr += 1 / (idx + 1)
                break
    return mrr / len(y_true)


def average_ranking(y_true, y_pred, k):
    assert len(y_true) == len(y_pred)
    average_ranking = 0.0
    for true, pred in zip(y_true, y_pred):
        for idx, item in enumerate(pred):
            if item == true:
                if idx <= k:
                    average_ranking += idx
                else:
                    average_ranking += k
                break
    return average_ranking / len(y_true)

import os
import pandas as pd
from tqdm import tqdm

def evaluate(model, testing_loader, k_values, reload_from_checkpoint=False, load_path_checkpoint=None, optimizer=None, data_path='/mnt/local/Baselines_Bugs/PatchSleuth/output/predict.csv'):
    device = configs.device
    
    if os.path.exists(data_path):
        os.rename(data_path, data_path+'.bak')

    output_df = pd.DataFrame(columns=['cve','output','label'])
    output_df.to_csv(data_path,index=False) 
    
    if reload_from_checkpoint:
        load_checkpoint(load_path_checkpoint, model)
    
    model.eval()
    cve_data = {}

    with torch.no_grad():
        for _, batch in tqdm(enumerate(testing_loader, 0), total=len(testing_loader)):
            label = batch['label'].cpu()
            input_ids_desc= batch['input_ids_desc'].to(device)
            attention_mask_desc= batch['attention_mask_desc'].to(device)
            input_ids_msg= batch['input_ids_msg'].to(device)
            attention_mask_msg= batch['attention_mask_msg'].to(device)
            input_ids_diff= batch['input_ids_diff'].to(device)
            attention_mask_diff= batch['attention_mask_diff'].to(device)
            model.to(device)
            output = model(input_ids_desc,input_ids_msg,attention_mask_desc,attention_mask_msg,input_ids_diff,attention_mask_diff)
            output.cpu()
            cve = batch['cve']

            for cve_i, output_i, label_i in zip(cve, output, label):
                cve_data.setdefault(cve_i, []).append((output_i.item(), label_i.item()))
                output_df = pd.DataFrame({'cve': [cve_i], 'output': [output_i.item()], 'label': [label_i.item()]})
                output_df.to_csv(data_path, mode='a', header=False, index=False)
        
        recalls = {k: [] for k in k_values}
        mrrs = []

        for cve_i, data in cve_data.items():
            data.sort(key=lambda x: x[0], reverse=True)
            ranks = [i for i, (_, label) in enumerate(data) if label == 1]

            for k in k_values:
                top_k_counts = sum(1 for rank in ranks if rank < k)
                recalls[k].append(top_k_counts / len(ranks) if ranks else 0)

            reciprocal_ranks = [1 / (rank + 1) for rank in ranks]
            mrrs.append(sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0)

        avg_recalls = {k: sum(recalls[k]) / len(recalls[k]) if recalls[k] else 0 for k in k_values}
        avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0

        for k in k_values:
            logging.info(f'Average Top@{k} recall: {avg_recalls[k]:.4f}')
        logging.info(f'Average MRR: {avg_mrr:.4f}')
        
        return avg_recalls, avg_mrr


#### average ranking algorithm
'''
if rank <= k:
    average_ranking += rank
else:
    average_ranking += k
'''





if __name__ == "__main__":
    # Modify paths and parameters as necessary:
    MODEL_PATH = "/mnt/local/Baselines_Bugs/PatchSleuth/model/output/Checkpoints/final_model.pt"  # modify this
    test_data = CVEDataset(configs.test_file)
    test_dataloader = DataLoader(test_data, batch_size=4, num_workers=10)
    model = CVEClassifier(
        lstm_hidden_size=256,
        num_classes=1,   # binary classification
        lstm_layers=1,
        dropout=0.1,
        lstm_input_size=512,  # Assuming a 512-sized embedding
        # lr=5e-5, 
        # num_train_epochs=20, 
        # warmup_steps=1000,
    )

    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100]

    # for model_path in model_paths:
    # model_path = MODEL_PATH
    model_path = '/mnt/local/Baselines_Bugs/PatchSleuth/model/output/Checkpoints/best-checkpoint.ckpt'
    logging.info(f'Evaluating model at {model_path}:')
    
    data_path_flag = model_path.split('/')[-1].split('.')[0]
    logging.info(f'data_path_flag: {data_path_flag}')
    recalls, avg_mrr = evaluate(
        model, 
        test_dataloader, 
        k_values, 
        reload_from_checkpoint=True,
        load_path_checkpoint=model_path,
        data_path=f'/mnt/local/Baselines_Bugs/PatchSleuth/output/predict_{data_path_flag}.csv'
        )
    

