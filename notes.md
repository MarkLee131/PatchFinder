help to check the metrics calculation logic in the two scripts:
`evaluate_new.py`
`# import os
import logging
import torch
import pandas as pd
# from collections import OrderedDict
from tqdm import tqdm
from torch.utils.data import DataLoader
import pytorch_lightning as pl
import torch.nn as nn
from transformers import AutoModelForSeq2SeqLM
import configs
from load_data_new import CVEDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class CVEClassifier(pl.LightningModule):
    def __init__(self, 
                 num_classes=1,
                 dropout=0.1,
                 lr=5e-5,
                 num_train_epochs=20,
                 warmup_steps=1000,
                 ):
        
        super().__init__()
        self.codeReviewer = AutoModelForSeq2SeqLM.from_pretrained(
            "microsoft/codereviewer").encoder
        
        self.save_hyperparameters()
        self.dropout = dropout
        self.criterion = nn.BCEWithLogitsLoss()

        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout)
        # Fully connected layer for output
        self.fc = nn.Linear(2 * self.codeReviewer.config.hidden_size, num_classes)

    def forward(self, input_ids_desc, attention_mask_desc, input_ids_msg_diff, attention_mask_msg_diff):
        
        # Get [CLS] embeddings for desc and msg+diff
        desc_cls_embed = self.codeReviewer(input_ids=input_ids_desc, attention_mask=attention_mask_desc).last_hidden_state[:, 0, :]
        msg_diff_cls_embed = self.codeReviewer(input_ids=input_ids_msg_diff, attention_mask=attention_mask_msg_diff).last_hidden_state[:, 0, :]
        
        # Concatenate [CLS] embeddings
        concatenated = torch.cat((desc_cls_embed, msg_diff_cls_embed), dim=1)
        
        # Apply dropout
        dropped = self.dropout_layer(concatenated)
        
        # Pass through the fully connected layer
        output = self.fc(dropped)
        
        return output
    
    def common_step(self, batch):
        predict = self(
            batch['input_ids_desc'],
            batch['attention_mask_desc'],
            batch['input_ids_msg_diff'],  # Updated to msg_diff
            batch['attention_mask_msg_diff']  # Updated to msg_diff
        )
        predict = predict.squeeze(1)
        loss = self.criterion(predict, batch['label'])
        return loss

    def training_step(self, batch, dataloader_idx=None):
        loss = self.common_step(batch)
        # logs metrics for each training_step,
        # and the average across the epoch
        self.log("training_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        loss = self.common_step(batch)
        self.log("validation_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx, dataloader_idx=None):
        loss = self.common_step(batch)

        return loss

    def test_dataloader(self):
        return test_dataloader


def load_checkpoint(load_path, model):
    # torch.save(model, model_save_path)
    model = torch.load(load_path)
    
    return model

def save_outputs_to_csv(cve, output, label, data_path):
    """Save model outputs to a CSV file."""
    output_df = pd.DataFrame({'cve': [cve], 'output': [output], 'label': [label]})
    output_df.to_csv(data_path, mode='a', header=False, index=False)

def evaluate_single_batch(model, batch):
    """Evaluate model on a single batch and return the output."""
    device = configs.device
    model.to(device)

    input_keys = ['input_ids_desc', 'attention_mask_desc', 'input_ids_msg_diff', 'attention_mask_msg_diff']
    inputs = [batch[key].to(device) for key in input_keys]

    return model(*inputs)

def compute_metrics(cve_data, k_values):
    """Compute metrics (recall@k, MRR, Manual Efforts) for the model outputs."""
    recalls = {k: [] for k in k_values}
    mrrs = []
    manual_efforts = {k: [] for k in k_values}

    for _, data in cve_data.items():
        data.sort(key=lambda x: x[0], reverse=True)
        ranks = [i for i, (_, label) in enumerate(data) if label == 1]

        for k in k_values:
            top_k_counts = sum(1 for rank in ranks if rank < k)
            recalls[k].append(top_k_counts / len(ranks) if ranks else 0)
            ### Manual Efforts
            '''
            if rank<k:
                manual_efforts[k].append(rank)
            else:
                manual_efforts[k].append(k)
            '''
            efforts_k = sum(rank if rank < k else k for rank in ranks)/len(ranks) if ranks else 0
            manual_efforts[k].append(efforts_k)

        reciprocal_ranks = [1 / (rank + 1) for rank in ranks]
        mrrs.append(sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0)

    avg_recalls = {k: sum(recalls[k]) / len(recalls[k]) if recalls[k] else 0 for k in k_values}
    avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0

    return avg_recalls, avg_mrr

def evaluate(model, testing_loader, k_values, reload_from_checkpoint=False, load_path_checkpoint=None, data_path='/mnt/local/Baselines_Bugs/PatchSleuth/output/predict.csv'):
    """Evaluate the model on the given test loader."""
    if reload_from_checkpoint:
        load_checkpoint(load_path_checkpoint, model)

    model.eval()
    cve_data = {}

    with torch.no_grad():
        for _, batch in tqdm(enumerate(testing_loader, 0), total=len(testing_loader)):
            output = evaluate_single_batch(model, batch)
            for cve_i, output_i, label_i in zip(batch['cve'], output, batch['label']):
                cve_data.setdefault(cve_i, []).append((output_i.item(), label_i.item()))
                save_outputs_to_csv(cve_i, output_i.item(), label_i.item(), data_path)
        
    avg_recalls, avg_mrr = compute_metrics(cve_data, k_values)

    for k in k_values:
        logging.info(f'Average Top@{k} recall: {avg_recalls[k]:.4f}')
    logging.info(f'Average MRR: {avg_mrr:.4f}')
    
    return avg_recalls, avg_mrr

if __name__ == "__main__":
    MODEL_PATH = "/mnt/local/Baselines_Bugs/PatchSleuth/model/output_0831/Checkpoints/final_model.pt"  
    test_data = CVEDataset(configs.test_file)
    test_dataloader = DataLoader(test_data, batch_size=8, num_workers=10)
    
    model = CVEClassifier(
            num_classes=1,   # binary classification
            dropout=0.1
        )
    k_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100]
    
    logging.info(f'Evaluating model at {MODEL_PATH}:')
    
    data_path_flag = MODEL_PATH.split('/')[-1].split('.')[0]
    logging.info(f'data_path_flag: {data_path_flag}')
    recalls, avg_mrr = evaluate(
        model, 
        test_dataloader, 
        k_values, 
        reload_from_checkpoint=True,
        load_path_checkpoint=MODEL_PATH,
        data_path=f'/mnt/local/Baselines_Bugs/PatchSleuth/model/output_0831/predict_{data_path_flag}.csv'
        )
`

`baseline_tfidf.py`
`### Used to calculate TF-IDF similarity between query and document first
# TF-IDF similarity
import os
from tqdm import tqdm
import pandas as pd

DATA_DIR = '/mnt/local/Baselines_Bugs/PatchSleuth/TF-IDF'
DATA_TMP_DIR = '/mnt/local/Baselines_Bugs/PatchSleuth/TF-IDF/tmp_0830'
os.makedirs(DATA_TMP_DIR, exist_ok=True)
    
### calculate the Top@K recall by using the rank info
def recall(k_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100], save_path=os.path.join(DATA_TMP_DIR, 'recall_TFIDF.csv')):
    recall_info = pd.DataFrame(columns=['Top@k', 'recall'])
    recall_info.to_csv(save_path, index=False)

    for k in k_list:
        rank_info = pd.read_csv(os.path.join(DATA_TMP_DIR, 'rankinfo_TDIDF.csv'))
        rank_info['recall'] = rank_info['rank'].apply(lambda x: 1 if x <= k else 0)
        recall = rank_info['recall'].sum() / len(rank_info)
        print('Top@{} recall: {}'.format(k, recall))
        recall_info_iter = pd.DataFrame([[k, recall]], columns=['Top@k', 'recall'])
        recall_info_iter.to_csv(save_path, mode='a', header=False, index=False)
        print('Top@{} recall info saved'.format(k))

### calculate the MRR
def mrr(save_path=os.path.join(DATA_TMP_DIR, 'MRR_TFIDF.csv')):
    rank_info = pd.read_csv(os.path.join(DATA_TMP_DIR, 'rankinfo_TDIDF.csv'))
    rank_info['reciprocal_rank'] = rank_info['rank'].apply(lambda x: 1 / x)
    mrr = rank_info['reciprocal_rank'].sum() / len(rank_info)
    print('MRR: {}'.format(mrr))
    mrr_info = pd.DataFrame([[mrr]], columns=['MRR'])
    mrr_info.to_csv(save_path, index=False)
    print('MRR info saved')
    
### calculate the average manual efforts for Top@K
'''
if rank <= k: 
    manual_efforts = rank
else: 
    manual_efforts = k
'''
def manual_efforts(k_list=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 50, 100], save_path=os.path.join(DATA_TMP_DIR, 'manualefforts_TFIDF.csv')):
    manual_efforts_info = pd.DataFrame(columns=['Top@k', 'manual_efforts'])
    manual_efforts_info.to_csv(save_path, index=False)
    
    for k in tqdm(k_list):
        rank_info = pd.read_csv(os.path.join(DATA_TMP_DIR, 'rankinfo_TDIDF.csv'))
        rank_info['manual_efforts'] = rank_info['rank'].apply(lambda x: x if x <= k else k)
        manual_efforts = rank_info['manual_efforts'].sum() / len(rank_info)
        print('Top@{} manual efforts: {}'.format(k, manual_efforts))
        manual_efforts_info_iter = pd.DataFrame([[k, manual_efforts]], columns=['Top@k', 'manual_efforts'])
        manual_efforts_info_iter.to_csv(save_path, mode='a', header=False, index=False)



if __name__ == "__main__":
    
    
    # ### step 0: preprocess the test data, just once, then use the saved csv file.
    # tf_idf_rawdata = pd.read_csv('/mnt/local/Baselines_Bugs/PatchSleuth/TF-IDF/similarity_data_TFIDF.csv')
    # test_data = pd.read_csv('/mnt/local/Baselines_Bugs/PatchSleuth/data/test_data.csv')
    # test_data.drop(['desc_token', 'msg_token', 'diff_token'], axis=1, inplace=True)
    # tfidf_data = pd.merge(tf_idf_rawdata, test_data, on=['cve', 'owner', 'repo', 'commit_id', 'label'], how='right')
    # tfidf_data.to_csv(os.path.join(DATA_DIR, 'test_data_TFIDF.csv'), index=False)
    
    ### read tfidf similarity data
    print("Step 1/3: read tfidf similarity data")
    tfidf_data = pd.read_csv('/mnt/local/Baselines_Bugs/PatchSleuth/TF-IDF/test_data_TFIDF.csv')
    tfidf_cve = tfidf_data.groupby('cve')

    print("Verify the number of cve:")
    cve_list = tfidf_data['cve'].unique().tolist()
    print('cve list length: {}'.format(len(cve_list)))
    cve_set = set(cve_list)
    print('cve set length: {}'.format(len(cve_set)))

    ### calculate the average rank of patch commit for each cve
    ### by using label to determine whether the commit is a patch commit
    print("Step 2/3: calculate the average rank of patch commit for each cve")
    rank_info = pd.DataFrame(columns=['cve', 'rank'])
    rank_info.to_csv(os.path.join(DATA_TMP_DIR, 'rankinfo_TDIDF.csv'), index=False)

    for cve, group in tqdm(tfidf_cve):
        # first sort the rows according to the similarity score
        group = group.sort_values(by='similarity', ascending=False)
        average_rank = 0
        #### maybe there are multiple patch commits for one cve
        patch_rows = group[group['label'] == 1]
        ranks = []
        for _, row in patch_rows.iterrows():
            ### get the rank by using the index of the row
            rank = group.index.get_loc(row.name) + 1
            ranks.append(rank)
        average_rank = sum(ranks) / len(ranks)
        rank_info_iter = pd.DataFrame([[cve, average_rank]], columns=['cve', 'rank'])
        rank_info_iter.to_csv(os.path.join(DATA_TMP_DIR, 'rankinfo_TDIDF.csv'), mode='a', header=False, index=False)
        # print('cve: {}, average rank: {}'.format(cve, average_rank))
    print('rank info saved')
    
    print("Step 3/3: calculate the recall, MRR and manual efforts")
    recall()
    mrr()
    manual_efforts()
    `
    note that I just care the correctness and consistency of the metrics calculation logic (recall, mrr, and manual_efforts).
