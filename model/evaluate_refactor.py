'''
NEED TO FIX!!!! Performance is not good as expected, maybe due to the wrong way of saving model predictions to CSV.

Refactored version of evaluate.py, which is used to evaluate the CR+LSTM model.(0830)
'''
# Standard Libraries
import os
from collections import OrderedDict

# Third-party Libraries
import torch
from torch import nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoModelForSeq2SeqLM
import pytorch_lightning as pl

# Custom Modules
import configs
from load_data import CVEDataset
from metrics import compute_metrics, save_metrics_to_csv, save_predictions_to_csv, print_metrics


MODEL_ROOT_DIR = '/mnt/local/Baselines_Bugs/PatchSleuth/metrics/CR_LSTM_0830'
# METRICS_DIR = os.path.join(MODEL_ROOT_DIR, 'metrics')
os.makedirs(MODEL_ROOT_DIR, exist_ok=True)
# os.makedirs(METRICS_DIR, exist_ok=True)


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


def evaluate(model, testing_loader, k_values, data_path):
    model.eval()
    cve_data = {}
    
    # Remove existing data file, since we are appending to it.
    if os.path.exists(data_path):
        os.rename(data_path, data_path + '.bak')

    with torch.no_grad():
        for _, batch in tqdm(enumerate(testing_loader, 0), total=len(testing_loader)):
            output = model(
                batch['input_ids_desc'],
                batch['attention_mask_desc'],
                batch['input_ids_msg'],
                batch['attention_mask_msg'],
                batch['input_ids_diff'],
                batch['attention_mask_diff']
            )
            cve = batch['cve']
            # print(f'cve: {cve}')
            output_list = output.squeeze(1).tolist()
            label = batch['label'].tolist()
            
            # Save model predictions to CSV
            save_predictions_to_csv(cve, output_list, label, data_path)
            
            for cve_i, output_i, label_i in zip(cve, output_list, label):
                cve_data.setdefault(cve_i, []).append((output_i, label_i))
    
    avg_recalls, avg_mrr, manual_efforts = compute_metrics(cve_data, k_values)
    print_metrics(avg_recalls, avg_mrr, manual_efforts, k_values)
    
    metrics_save_path = os.path.join(MODEL_ROOT_DIR, f'metrics_{data_path_flag}.csv')
    save_metrics_to_csv(avg_recalls, avg_mrr, manual_efforts, metrics_save_path)
    
    return avg_recalls, avg_mrr

if __name__ == "__main__":
    
    model_path = os.path.join(MODEL_ROOT_DIR, 'Checkpoints', 'final_model.pt') 
    test_data = CVEDataset(configs.test_file)
    test_dataloader = DataLoader(test_data, batch_size=4, num_workers=10)

    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100]
    print(f'Evaluating model at {model_path}')
    
    model = CVEClassifier(
        lstm_hidden_size=256,
        num_classes=1,
        lstm_layers=1,
        dropout=0.1,
        lstm_input_size=512,
    )
    model.load_state_dict(torch.load(model_path))
    
    data_path_flag = model_path.split('/')[-1].split('.')[0]
    print(f'data_path_flag: {data_path_flag}')
    recalls, avg_mrr = evaluate(
        model, 
        test_dataloader, 
        k_values, 
        data_path=os.path.join(MODEL_ROOT_DIR, f'predict_{data_path_flag}.csv')
    )
